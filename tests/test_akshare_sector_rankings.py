# -*- coding: utf-8 -*-
"""Tests for AkshareFetcher.get_sector_rankings board-internals passthrough.

板块榜单接口本身就返回涨跌家数与领涨个股，这些字段用于生成板块异动原因。
本文件锁定契约：

- 东财源透传 up_count / down_count / leader_stock / leader_change_pct
- 新浪备用源透传 member_count / leader_stock / leader_change_pct
- 缺列、空值时只省略对应字段，name / change_pct 始终保留
- 排序仍按涨跌幅取 nlargest / nsmallest

Covers: data_provider/akshare_fetcher.py get_sector_rankings
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

from data_provider.akshare_fetcher import AkshareFetcher


def _make_em_df():
    """Simulate ak.stock_board_industry_name_em() return value."""
    return pd.DataFrame([
        {
            '排名': 1, '板块名称': '煤炭行业', '板块代码': 'BK0437', '涨跌幅': 5.22,
            '上涨家数': 35, '下跌家数': 0, '领涨股票': '云煤能源', '领涨股票-涨跌幅': 10.12,
        },
        {
            '排名': 2, '板块名称': '互联网服务', '板块代码': 'BK0447', '涨跌幅': 4.13,
            '上涨家数': 144, '下跌家数': 3, '领涨股票': '信雅达', '领涨股票-涨跌幅': 9.97,
        },
        {
            '排名': 85, '板块名称': '银行', '板块代码': 'BK0475', '涨跌幅': -1.30,
            '上涨家数': 2, '下跌家数': 38, '领涨股票': '成都银行', '领涨股票-涨跌幅': 0.45,
        },
        {
            '排名': 86, '板块名称': '珠宝首饰', '板块代码': 'BK0734', '涨跌幅': -2.05,
            '上涨家数': 1, '下跌家数': 8, '领涨股票': '曼卡龙', '领涨股票-涨跌幅': 1.36,
        },
    ])


def _make_sina_df():
    """Simulate ak.stock_sector_spot(indicator='行业') return value."""
    return pd.DataFrame([
        {
            'label': 'new_dlhy', '板块': '煤炭行业', '公司家数': 37, '涨跌幅': 5.22,
            '股票代码': '600792', '个股-涨跌幅': 10.12, '股票名称': '云煤能源',
        },
        {
            'label': 'new_yh', '板块': '银行', '公司家数': 42, '涨跌幅': -1.30,
            '股票代码': '601838', '个股-涨跌幅': 0.45, '股票名称': '成都银行',
        },
    ])


class TestAkshareSectorRankings(unittest.TestCase):
    def setUp(self):
        self.fetcher = AkshareFetcher()
        # Bypass rate limiting / UA rotation
        self.fetcher._enforce_rate_limit = lambda: None
        self.fetcher._set_random_user_agent = lambda: None

    def test_eastmoney_passes_through_board_internals(self):
        with patch('akshare.stock_board_industry_name_em', return_value=_make_em_df()):
            top, bottom = self.fetcher.get_sector_rankings(2)

        self.assertEqual([s['name'] for s in top], ['煤炭行业', '互联网服务'])
        self.assertEqual([s['name'] for s in bottom], ['珠宝首饰', '银行'])
        self.assertEqual(top[0], {
            'name': '煤炭行业',
            'change_pct': 5.22,
            'up_count': 35,
            'down_count': 0,
            'leader_stock': '云煤能源',
            'leader_change_pct': 10.12,
        })
        # 领跌板块同样带内部结构，异动原因才能说明是普跌还是分化
        self.assertEqual(bottom[1]['up_count'], 2)
        self.assertEqual(bottom[1]['down_count'], 38)
        self.assertIsInstance(top[0]['up_count'], int)

    def test_sina_fallback_passes_through_available_internals(self):
        with patch('akshare.stock_board_industry_name_em', side_effect=RuntimeError('em down')), \
                patch('akshare.stock_sector_spot', return_value=_make_sina_df()):
            top, bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0], {
            'name': '煤炭行业',
            'change_pct': 5.22,
            'member_count': 37,
            'leader_stock': '云煤能源',
            'leader_change_pct': 10.12,
        })
        # 新浪源没有涨跌家数，不应凭空补字段
        self.assertNotIn('up_count', top[0])
        self.assertEqual(bottom[0]['name'], '银行')

    def test_missing_internal_columns_keep_base_contract(self):
        df = _make_em_df()[['板块名称', '涨跌幅']]

        with patch('akshare.stock_board_industry_name_em', return_value=df):
            top, bottom = self.fetcher.get_sector_rankings(1)

        self.assertEqual(top[0], {'name': '煤炭行业', 'change_pct': 5.22})
        self.assertEqual(bottom[0], {'name': '珠宝首饰', 'change_pct': -2.05})

    def test_null_internal_values_are_skipped(self):
        df = _make_em_df()
        df.loc[0, '领涨股票'] = None
        df.loc[0, '上涨家数'] = None

        with patch('akshare.stock_board_industry_name_em', return_value=df):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertNotIn('leader_stock', top[0])
        self.assertNotIn('up_count', top[0])
        self.assertEqual(top[0]['down_count'], 0)
        self.assertEqual(top[0]['leader_change_pct'], 10.12)

    def test_blank_leader_name_is_skipped(self):
        df = _make_em_df()
        df.loc[0, '领涨股票'] = '   '

        with patch('akshare.stock_board_industry_name_em', return_value=df):
            top, _bottom = self.fetcher.get_sector_rankings(1)

        self.assertNotIn('leader_stock', top[0])

    def test_returns_none_when_all_sources_fail(self):
        with patch('akshare.stock_board_industry_name_em', side_effect=RuntimeError('em down')), \
                patch('akshare.stock_sector_spot', side_effect=RuntimeError('sina down')):
            self.assertIsNone(self.fetcher.get_sector_rankings(5))


if __name__ == '__main__':
    unittest.main()
