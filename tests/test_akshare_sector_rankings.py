# -*- coding: utf-8 -*-
"""Tests for AkshareFetcher industry ranking source priority and catalyst material.

行业榜单口径为申万一级（31 个），与券商盘后复盘一致；东财二级 / 新浪仅作降级。
本文件锁定契约：

- 数据源优先级 申万一级 -> 东财二级 -> 新浪，每级都带 taxonomy 标记
- 申万接口只给点位，涨跌幅必须由 (最新价-昨收盘)/昨收盘 计算
- 缺列、空值时只省略对应字段，name / change_pct 始终保留
- 催化材料：子板块按上级行业归属（不靠代码前缀）、涨停股交叉成分股、领涨领跌个股

Covers: data_provider/akshare_fetcher.py get_sector_rankings / get_sector_catalyst_context
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()
try:
    import json_repair  # noqa: F401
except ImportError:
    if "json_repair" not in sys.modules:
        sys.modules["json_repair"] = MagicMock()

import data_provider.akshare_fetcher as akshare_fetcher_module
from data_provider.akshare_fetcher import AkshareFetcher


def _reset_market_spot_cache():
    """全市场行情缓存是模块级共享状态（供各数据源复用），测试间必须清空以保证隔离。"""
    akshare_fetcher_module._realtime_cache['data'] = None
    akshare_fetcher_module._realtime_cache['timestamp'] = 0


def _sw_level1_df():
    """Simulate ak.index_realtime_sw(symbol='一级行业')：只给点位，无涨跌幅。"""
    return pd.DataFrame([
        {'指数代码': '801950', '指数名称': '煤炭', '昨收盘': 1000.0, '最新价': 1010.40},
        {'指数代码': '801130', '指数名称': '纺织服饰', '昨收盘': 1000.0, '最新价': 1005.20},
        {'指数代码': '801050', '指数名称': '有色金属', '昨收盘': 1000.0, '最新价': 972.10},
        {'指数代码': '801710', '指数名称': '建筑材料', '昨收盘': 1000.0, '最新价': 975.10},
    ])


def _em_level2_df():
    return pd.DataFrame([
        {'板块名称': '煤炭行业', '涨跌幅': 5.22, '上涨家数': 35, '下跌家数': 0,
         '领涨股票': '云煤能源', '领涨股票-涨跌幅': 10.12},
        {'板块名称': '珠宝首饰', '涨跌幅': -2.05, '上涨家数': 1, '下跌家数': 8,
         '领涨股票': '曼卡龙', '领涨股票-涨跌幅': 1.36},
    ])


def _sina_df():
    return pd.DataFrame([
        {'板块': '煤炭行业', '公司家数': 37, '涨跌幅': 5.22,
         '股票名称': '云煤能源', '个股-涨跌幅': 10.12},
        {'板块': '银行', '公司家数': 42, '涨跌幅': -1.30,
         '股票名称': '成都银行', '个股-涨跌幅': 0.45},
    ])


class TestSectorRankingSourcePriority(unittest.TestCase):
    def setUp(self):
        self.fetcher = AkshareFetcher()
        self.fetcher._enforce_rate_limit = lambda: None
        self.fetcher._set_random_user_agent = lambda: None

    def test_prefers_sw_level1_and_computes_change_pct(self):
        with patch('akshare.index_realtime_sw', return_value=_sw_level1_df()) as sw, \
                patch('akshare.stock_board_industry_name_em') as em:
            top, bottom = self.fetcher.get_sector_rankings(2)

        sw.assert_called_once_with(symbol='一级行业')
        em.assert_not_called()  # 申万成功则不应触碰东财

        self.assertEqual([s['name'] for s in top], ['煤炭', '纺织服饰'])
        self.assertEqual([s['name'] for s in bottom], ['有色金属', '建筑材料'])
        self.assertAlmostEqual(top[0]['change_pct'], 1.04, places=2)
        self.assertAlmostEqual(bottom[0]['change_pct'], -2.79, places=2)
        self.assertEqual(top[0]['taxonomy'], 'sw_l1')
        self.assertEqual(top[0]['code'], '801950')

    def test_skips_zero_previous_close(self):
        df = _sw_level1_df()
        df.loc[0, '昨收盘'] = 0.0

        with patch('akshare.index_realtime_sw', return_value=df), \
                patch('akshare.stock_board_industry_name_em'):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0]['name'], '纺织服饰')

    def test_degrades_to_eastmoney_when_sw_missing_columns(self):
        with patch('akshare.index_realtime_sw', return_value=pd.DataFrame([{'foo': 1}])), \
                patch('akshare.stock_board_industry_name_em', return_value=_em_level2_df()):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0]['name'], '煤炭行业')
        self.assertEqual(top[0]['taxonomy'], 'em_l2')
        self.assertEqual(top[0]['up_count'], 35)
        self.assertEqual(top[0]['leader_stock'], '云煤能源')

    def test_degrades_to_eastmoney_when_sw_raises(self):
        with patch('akshare.index_realtime_sw', side_effect=RuntimeError('sw down')), \
                patch('akshare.stock_board_industry_name_em', return_value=_em_level2_df()):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0]['taxonomy'], 'em_l2')

    def test_degrades_to_sina_when_both_fail(self):
        with patch('akshare.index_realtime_sw', side_effect=RuntimeError('sw down')), \
                patch('akshare.stock_board_industry_name_em', side_effect=RuntimeError('em down')), \
                patch('akshare.stock_sector_spot', return_value=_sina_df()):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0]['taxonomy'], 'sina')
        self.assertEqual(top[0]['member_count'], 37)
        self.assertNotIn('up_count', top[0])

    def test_returns_none_when_all_sources_fail(self):
        with patch('akshare.index_realtime_sw', side_effect=RuntimeError('sw down')), \
                patch('akshare.stock_board_industry_name_em', side_effect=RuntimeError('em down')), \
                patch('akshare.stock_sector_spot', side_effect=RuntimeError('sina down')):
            self.assertIsNone(self.fetcher.get_sector_rankings(3))

    def test_missing_internal_columns_keep_base_contract(self):
        df = _em_level2_df()[['板块名称', '涨跌幅']]

        with patch('akshare.index_realtime_sw', side_effect=RuntimeError('sw down')), \
                patch('akshare.stock_board_industry_name_em', return_value=df):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0], {'name': '煤炭行业', 'change_pct': 5.22, 'taxonomy': 'em_l2'})

    def test_null_internal_values_are_skipped(self):
        df = _em_level2_df()
        df.loc[0, '领涨股票'] = None
        df.loc[0, '上涨家数'] = None

        with patch('akshare.index_realtime_sw', side_effect=RuntimeError('sw down')), \
                patch('akshare.stock_board_industry_name_em', return_value=df):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertNotIn('leader_stock', top[0])
        self.assertNotIn('up_count', top[0])
        self.assertEqual(top[0]['down_count'], 0)


class TestSectorCatalystContext(unittest.TestCase):
    def setUp(self):
        self.fetcher = AkshareFetcher()
        self.fetcher._enforce_rate_limit = lambda: None
        self.fetcher._set_random_user_agent = lambda: None
        self.sectors = [{'name': '煤炭', 'code': '801950', 'taxonomy': 'sw_l1'}]
        _reset_market_spot_cache()
        self.addCleanup(_reset_market_spot_cache)

    @staticmethod
    def _level2_spot():
        return pd.DataFrame([
            {'指数名称': '焦炭Ⅱ', '昨收盘': 1000.0, '最新价': 1020.80},
            {'指数名称': '煤炭开采', '昨收盘': 1000.0, '最新价': 1007.80},
            {'指数名称': '玻璃玻纤', '昨收盘': 1000.0, '最新价': 968.20},
        ])

    @staticmethod
    def _level2_info():
        return pd.DataFrame([
            {'行业代码': '801951.SI', '行业名称': '焦炭Ⅱ', '上级行业': '煤炭'},
            {'行业代码': '801952.SI', '行业名称': '煤炭开采', '上级行业': '煤炭'},
            {'行业代码': '801712.SI', '行业名称': '玻璃玻纤', '上级行业': '建筑材料'},
        ])

    @staticmethod
    def _zt_pool():
        return pd.DataFrame([
            {'代码': '600792', '名称': '云煤能源', '涨跌幅': 10.02,
             '连板数': 2, '所属行业': '煤炭开采'},
            {'代码': '001234', '名称': '泰慕士', '涨跌幅': 10.02,
             '连板数': 3, '所属行业': '服装家纺'},
        ])

    @staticmethod
    def _components():
        return pd.DataFrame([
            {'证券代码': '600792', '证券名称': '云煤能源'},
            {'证券代码': '601666', '证券名称': '平煤股份'},
        ])

    @staticmethod
    def _spot():
        return pd.DataFrame([
            {'代码': '600792', '名称': '云煤能源', '涨跌幅': 10.02},
            {'代码': '601666', '名称': '平煤股份', '涨跌幅': -1.20},
            {'代码': '001234', '名称': '泰慕士', '涨跌幅': 10.02},
        ])

    def test_collects_sub_sectors_limit_ups_and_movers(self):
        with patch('akshare.index_realtime_sw', return_value=self._level2_spot()), \
                patch('akshare.sw_index_second_info', return_value=self._level2_info()), \
                patch('akshare.stock_zt_pool_em', return_value=self._zt_pool()), \
                patch('akshare.index_component_sw', return_value=self._components()), \
                patch('akshare.stock_zh_a_spot_em', return_value=self._spot()):
            ctx = self.fetcher.get_sector_catalyst_context(self.sectors, date='20260923')

        material = ctx['煤炭']
        # 子板块按上级行业归属，且只含本行业的，降序排列
        self.assertEqual([s['name'] for s in material['sub_sectors']], ['焦炭Ⅱ', '煤炭开采'])
        self.assertAlmostEqual(material['sub_sectors'][0]['change_pct'], 2.08, places=2)
        # 涨停股按成分股代码精确归因，不会把别的行业的带进来
        self.assertEqual([s['name'] for s in material['limit_up_stocks']], ['云煤能源'])
        self.assertEqual(material['limit_up_stocks'][0]['streak'], 2)
        self.assertEqual(material['limit_up_stocks'][0]['sub_industry'], '煤炭开采')
        self.assertEqual(material['top_gainers'][0]['name'], '云煤能源')
        self.assertEqual(material['top_losers'][0]['name'], '平煤股份')

    def test_degrades_to_sina_spot_when_eastmoney_spot_fails(self):
        sina_spot = pd.DataFrame([
            {'代码': 'sh600792', '名称': '云煤能源', '涨跌幅': 10.02},
            {'代码': 'sh601666', '名称': '平煤股份', '涨跌幅': -1.20},
        ])

        with patch('akshare.index_realtime_sw', return_value=self._level2_spot()), \
                patch('akshare.sw_index_second_info', return_value=self._level2_info()), \
                patch('akshare.stock_zt_pool_em', return_value=self._zt_pool()), \
                patch('akshare.index_component_sw', return_value=self._components()), \
                patch('akshare.stock_zh_a_spot_em', side_effect=RuntimeError('em down')), \
                patch('akshare.stock_zh_a_spot', return_value=sina_spot):
            ctx = self.fetcher.get_sector_catalyst_context(self.sectors, date='20260923')

        # 新浪代码带 sh/sz 前缀，必须归一化后才能匹配成分股
        self.assertEqual(ctx['煤炭']['top_gainers'][0]['name'], '云煤能源')

    def test_survives_every_material_source_failing(self):
        with patch('akshare.index_realtime_sw', side_effect=RuntimeError('down')), \
                patch('akshare.sw_index_second_info', side_effect=RuntimeError('down')), \
                patch('akshare.stock_zt_pool_em', side_effect=RuntimeError('down')), \
                patch('akshare.stock_zh_a_spot_em', side_effect=RuntimeError('down')), \
                patch('akshare.stock_zh_a_spot', side_effect=RuntimeError('down')):
            ctx = self.fetcher.get_sector_catalyst_context(self.sectors, date='20260923')

        self.assertEqual(ctx['煤炭'], {
            'sub_sectors': [], 'limit_up_stocks': [], 'top_gainers': [], 'top_losers': [],
        })

    def test_skips_constituent_lookup_for_non_sw_taxonomy(self):
        """东财二级降级口径没有申万指数代码，不应去查成分股。"""
        with patch('akshare.index_realtime_sw', return_value=self._level2_spot()), \
                patch('akshare.sw_index_second_info', return_value=self._level2_info()), \
                patch('akshare.stock_zt_pool_em', return_value=self._zt_pool()), \
                patch('akshare.stock_zh_a_spot_em', return_value=self._spot()), \
                patch('akshare.index_component_sw') as comp:
            ctx = self.fetcher.get_sector_catalyst_context(
                [{'name': '煤炭行业', 'taxonomy': 'em_l2'}], date='20260923'
            )

        comp.assert_not_called()
        self.assertEqual(ctx['煤炭行业']['limit_up_stocks'], [])

    def test_returns_empty_for_no_sectors(self):
        self.assertEqual(self.fetcher.get_sector_catalyst_context([]), {})


if __name__ == '__main__':
    unittest.main()


class TestSectorNewsSources(unittest.TestCase):
    """东财板块新闻 / 财联社电报：通用搜索未配置时的催化来源，无需任何 API Key。"""

    def setUp(self):
        self.fetcher = AkshareFetcher()
        self.fetcher._enforce_rate_limit = lambda: None
        self.fetcher._set_random_user_agent = lambda: None

    @staticmethod
    def _news_df():
        return pd.DataFrame([
            {'新闻标题': '煤炭采选板块震荡反弹 云煤能源直线涨停',
             '新闻内容': '消息面上，生意社动力煤基准价989.50元/吨，较本月初上涨13.02%。',
             '发布时间': '2026-09-24 09:41:48', '文章来源': '东方财富Choice数据'},
            # 以下四条都是纯涨跌幅复述的榜单稿，必须被过滤
            {'新闻标题': '云煤能源600792龙虎榜数据09-22', '新闻内容': '当日收报5.75元，涨跌幅5.31%。',
             '发布时间': '2026-09-22 16:58:52', '文章来源': '东方财富Choice数据'},
            {'新闻标题': '39只股上午收盘涨停(附股)', '新闻内容': '601567 三星电气 16.25 ...',
             '发布时间': '2026-09-24 11:35:00', '文章来源': '证券时报网'},
            {'新闻标题': '今日沪指跌0.93% 有色金属行业跌幅最大', '新闻内容': '从申万行业来看...',
             '发布时间': '2026-09-24 13:14:00', '文章来源': '证券时报网'},
            {'新闻标题': '某公司公告', '新闻内容': '本文基于AI生产，仅供参考',
             '发布时间': '2026-09-24 10:00:00', '文章来源': 'AI'},
            {'新闻标题': '煤炭保供贵在安全主动', '新闻内容': '在关键时间窗口发力稳煤保供。',
             '发布时间': '2026-09-24 07:57:40', '文章来源': '经济日报'},
        ])

    def test_fetches_news_by_industry_name_and_filters_noise(self):
        with patch('akshare.stock_news_em', return_value=self._news_df()) as m:
            result = self.fetcher.get_sector_news(['煤炭'], max_items=5)

        m.assert_called_once_with(symbol='煤炭')
        titles = [item['title'] for item in result['煤炭']]
        self.assertEqual(titles, ['煤炭采选板块震荡反弹 云煤能源直线涨停', '煤炭保供贵在安全主动'])
        self.assertIn('13.02%', result['煤炭'][0]['snippet'])
        self.assertEqual(result['煤炭'][0]['source'], '东方财富Choice数据')

    def test_respects_max_items_and_dedupes_names(self):
        with patch('akshare.stock_news_em', return_value=self._news_df()) as m:
            result = self.fetcher.get_sector_news(['煤炭', '煤炭', ''], max_items=1)

        self.assertEqual(m.call_count, 1)
        self.assertEqual(len(result['煤炭']), 1)

    def test_isolates_per_industry_failure(self):
        def side_effect(symbol):
            if symbol == '煤炭':
                raise RuntimeError('news down')
            return self._news_df()

        with patch('akshare.stock_news_em', side_effect=side_effect):
            result = self.fetcher.get_sector_news(['煤炭', '银行'])

        self.assertNotIn('煤炭', result)
        self.assertIn('银行', result)

    def test_returns_empty_when_all_news_filtered(self):
        df = self._news_df().iloc[[1, 2, 3, 4]]  # 全是噪声行

        with patch('akshare.stock_news_em', return_value=df):
            self.assertEqual(self.fetcher.get_sector_news(['煤炭']), {})

    def test_market_wire_news(self):
        df = pd.DataFrame([
            {'标题': '高盛：预计美联储10月完成最后一次加息', '内容': '财联社9月24日电，高盛在最新报告中...',
             '发布日期': '2026-09-24', '发布时间': '12:30:00'},
            {'标题': '', '内容': '台交所加权股价指数收低0.3%报48,024.60点。',
             '发布日期': '2026-09-24', '发布时间': '13:40:00'},
        ])

        with patch('akshare.stock_info_global_cls', return_value=df) as m:
            items = self.fetcher.get_market_wire_news(limit=5)

        m.assert_called_once_with(symbol='全部')
        self.assertEqual(items[0]['source'], '财联社')
        self.assertEqual(items[0]['published_date'], '2026-09-24 12:30:00')
        # 无标题时用正文开头兜底，避免整条丢失
        self.assertTrue(items[1]['title'].startswith('台交所'))

    def test_market_wire_news_survives_failure(self):
        with patch('akshare.stock_info_global_cls', side_effect=RuntimeError('cls down')):
            self.assertEqual(self.fetcher.get_market_wire_news(), [])
