# -*- coding: utf-8 -*-
"""Tests for the industry driver (行业异动原因) enrichment.

Covers:
- 解读层：LLM 输出解析、噪声容忍、无依据/未知/歧义行业名的丢弃、长度截断、表格安全
- 归因层：模型弃权时只做子板块归因，且不得复述涨跌幅或涨跌家数
- 降级路径：未配置模型 / 开关关闭 / 无任何新闻 / 调用抛错 / 返回 None
- Prompt 契约：行业材料（子板块、涨停股、领涨领跌个股、行业新闻）必须进入 prompt
- 集成：表格 Top3、涨幅/跌幅 表头、按行业检索新闻的调用与失败隔离
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Stub heavy optional dependencies before project imports
for _mod in ("litellm", "google.generativeai", "google.genai", "anthropic"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest

from src.services.sector_driver_service import SectorDriverService


def _analyzer(response=None, *, available=True, raises=None):
    """Minimal analyzer double exposing only the public generate_text contract."""

    def generate_text(prompt, max_tokens=None, temperature=None):
        generate_text.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if raises is not None:
            raise raises
        return response

    generate_text.calls = []
    return SimpleNamespace(generate_text=generate_text, is_available=lambda: available)


def _news(title="发改委部署迎峰度夏能源保供"):
    return [{"title": title, "snippet": "要求主产区稳产增产", "source": "新华社"}]


def _context(name="煤炭"):
    """典型的一份行业材料，结构与 AkshareFetcher.get_sector_catalyst_context 一致。"""
    return {
        name: {
            "sub_sectors": [
                {"name": "焦炭Ⅱ", "change_pct": 2.08},
                {"name": "煤炭开采", "change_pct": 0.78},
            ],
            "limit_up_stocks": [
                {"name": "云煤能源", "change_pct": 10.02, "sub_industry": "煤炭开采", "streak": 2},
            ],
            "top_gainers": [{"name": "云煤能源", "change_pct": 10.02}],
            "top_losers": [{"name": "平煤股份", "change_pct": -1.20}],
        }
    }


class TestAttributionLayer:
    """归因层：模型不可用时只回答「行业内部哪一块在动」，不复述涨跌幅。"""

    def test_leading_industry_attributes_to_strongest_sub_sector(self):
        top = [{"name": "煤炭", "change_pct": 1.04}]

        stats = SectorDriverService().annotate(top, [], catalyst_context=_context())

        assert top[0]["reason"] == (
            "主要由焦炭Ⅱ(+2.08%)带动，云煤能源等涨停；未检索到明确消息催化"
        )
        assert top[0]["reason_source"] == "sub_sector_attribution"
        assert stats == {"total": 1, "llm": 0, "attribution": 1, "empty": 0}

    def test_lagging_industry_attributes_to_weakest_sub_sector(self):
        bottom = [{"name": "建筑材料", "change_pct": -2.49}]
        ctx = {
            "建筑材料": {
                "sub_sectors": [
                    {"name": "装修建材", "change_pct": -1.61},
                    {"name": "水泥", "change_pct": -1.97},
                    {"name": "玻璃玻纤", "change_pct": -3.18},
                ],
            }
        }

        SectorDriverService().annotate([], bottom, catalyst_context=ctx)

        assert bottom[0]["reason"] == "主要由玻璃玻纤(-3.18%)拖累；未检索到明确消息催化"

    def test_lagging_industry_does_not_list_limit_up_stocks(self):
        """领跌行业里零星涨停股不是异动原因，不应混进来。"""
        bottom = [{"name": "煤炭", "change_pct": -1.0}]

        SectorDriverService().annotate([], bottom, catalyst_context=_context())

        assert "涨停" not in bottom[0]["reason"]
        assert "煤炭开采" in bottom[0]["reason"]

    def test_never_restates_breadth_or_change_pct(self):
        """回归用例：旧实现会输出「35涨0跌，龙头xx +10%」，这类文案不是异动原因。"""
        top = [{
            "name": "煤炭", "change_pct": 1.04,
            "up_count": 35, "down_count": 0,
            "leader_stock": "云煤能源", "leader_change_pct": 10.12,
        }]

        SectorDriverService().annotate(top, [], catalyst_context=_context())

        reason = top[0]["reason"]
        assert "涨0跌" not in reason
        assert "龙头" not in reason
        assert "板块普涨" not in reason

    def test_leaves_reason_absent_without_any_material(self):
        top = [{"name": "煤炭", "change_pct": 1.04}]

        stats = SectorDriverService().annotate(top, [], catalyst_context={})

        assert "reason" not in top[0]
        assert stats == {"total": 1, "llm": 0, "attribution": 0, "empty": 1}

    def test_english_attribution(self):
        bottom = [{"name": "Building materials", "change_pct": -2.49}]
        ctx = {"Building materials": {"sub_sectors": [{"name": "Glass fiber", "change_pct": -3.18}]}}

        SectorDriverService().annotate([], bottom, catalyst_context=ctx, language="en")

        assert bottom[0]["reason"] == (
            "mainly dragged down by Glass fiber (-3.18%); no clear news catalyst found"
        )


class TestLlmLayer:
    def test_parses_and_prefers_llm_reason(self):
        analyzer = _analyzer(
            "煤炭 => 发改委保供表态叠加长协价机制优化，焦炭Ⅱ领涨，港口库存降至近三年低位"
        )
        top = [{"name": "煤炭", "change_pct": 1.04}]

        stats = SectorDriverService(analyzer=analyzer).annotate(
            top, [], news=_news(), catalyst_context=_context()
        )

        assert top[0]["reason"].startswith("发改委保供表态")
        assert top[0]["reason_source"] == "llm"
        assert stats == {"total": 1, "llm": 1, "attribution": 0, "empty": 0}

    def test_prompt_carries_all_industry_material(self):
        analyzer = _analyzer("煤炭 => 保供预期升温")
        top = [{"name": "煤炭", "change_pct": 1.04}]

        SectorDriverService(analyzer=analyzer).annotate(
            top, [],
            news=[{"title": "市场级新闻标题", "snippet": "摘要", "source": "证券时报"}],
            sector_news={"煤炭": _news("煤炭行业专属新闻")},
            catalyst_context=_context(),
        )

        prompt = analyzer.generate_text.calls[0]["prompt"]
        assert "领涨 煤炭 +1.04%" in prompt
        assert "焦炭Ⅱ +2.08%" in prompt                 # 子板块
        assert "云煤能源(煤炭开采/2板)" in prompt          # 涨停股含子行业与连板数
        assert "平煤股份 -1.20%" in prompt                # 领跌个股
        assert "煤炭行业专属新闻" in prompt               # 按行业检索的新闻
        assert "市场级新闻标题" in prompt                 # 市场级新闻
        # 反「复述涨跌幅」的硬约束必须在 prompt 里
        assert "禁止只复述涨跌幅" in prompt

    def test_tolerates_bullets_numbering_and_fences(self):
        analyzer = _analyzer(
            "```\n1. 煤炭 (+1.04%)：发改委保供增产\n- 建筑材料 → 地产需求疲软\n```"
        )
        top = [{"name": "煤炭", "change_pct": 1.04}]
        bottom = [{"name": "建筑材料", "change_pct": -2.49}]

        SectorDriverService(analyzer=analyzer).annotate(
            top, bottom, news=_news(), catalyst_context=_context()
        )

        assert top[0]["reason"] == "发改委保供增产"
        assert bottom[0]["reason"] == "地产需求疲软"

    @pytest.mark.parametrize("token", ["无", "none", "N/A", "暂无数据", "-", "弃权"])
    def test_falls_back_when_model_abstains(self, token):
        analyzer = _analyzer(f"煤炭 => {token}")
        top = [{"name": "煤炭", "change_pct": 1.04}]

        SectorDriverService(analyzer=analyzer).annotate(
            top, [], news=_news(), catalyst_context=_context()
        )

        assert top[0]["reason_source"] == "sub_sector_attribution"
        assert "焦炭Ⅱ" in top[0]["reason"]

    def test_drops_unknown_industry_name(self):
        analyzer = _analyzer("白酒 => 春节备货超预期")
        top = [{"name": "煤炭", "change_pct": 1.04}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert "reason" not in top[0]

    def test_drops_ambiguous_industry_name(self):
        analyzer = _analyzer("银行 => 高股息受青睐")
        top = [{"name": "国有大型银行", "change_pct": 1.0}, {"name": "城商行银行", "change_pct": 0.9}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert "reason" not in top[0]
        assert "reason" not in top[1]

    def test_truncates_at_120_chars(self):
        analyzer = _analyzer(f"煤炭 => {'政策预期升温' * 40}")
        top = [{"name": "煤炭", "change_pct": 1.04}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert len(top[0]["reason"]) == SectorDriverService.MAX_REASON_CHARS == 120
        assert top[0]["reason"].endswith("...")

    def test_strips_pipe_to_keep_table_intact(self):
        analyzer = _analyzer("煤炭 => 保供增产 | 长协优化")
        top = [{"name": "煤炭", "change_pct": 1.04}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert "|" not in top[0]["reason"]
        assert top[0]["reason"] == "保供增产 / 长协优化"


class TestDegradation:
    def _top(self):
        return [{"name": "煤炭", "change_pct": 1.04}]

    def test_sector_news_alone_is_enough_to_call_llm(self):
        """只有行业新闻、没有市场级新闻时也应调用模型。"""
        analyzer = _analyzer("煤炭 => 保供预期升温")
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(
            top, [], news=[], sector_news={"煤炭": _news()}
        )

        assert analyzer.generate_text.calls
        assert top[0]["reason_source"] == "llm"

    def test_skips_llm_without_any_news(self):
        analyzer = _analyzer("煤炭 => 不该被调用")
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(
            top, [], news=[], sector_news={}, catalyst_context=_context()
        )

        assert analyzer.generate_text.calls == []
        assert top[0]["reason_source"] == "sub_sector_attribution"

    def test_skips_llm_when_config_disables_it(self):
        analyzer = _analyzer("煤炭 => 不该被调用")
        top = self._top()

        SectorDriverService(
            config=SimpleNamespace(market_sector_reason_enabled=False), analyzer=analyzer
        ).annotate(top, [], news=_news(), catalyst_context=_context())

        assert analyzer.generate_text.calls == []
        assert top[0]["reason_source"] == "sub_sector_attribution"

    def test_skips_llm_when_analyzer_unavailable(self):
        analyzer = _analyzer("煤炭 => 不该被调用", available=False)
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(
            top, [], news=_news(), catalyst_context=_context()
        )

        assert analyzer.generate_text.calls == []

    def test_falls_back_when_generate_text_raises(self):
        top = self._top()

        SectorDriverService(analyzer=_analyzer(raises=RuntimeError("llm down"))).annotate(
            top, [], news=_news(), catalyst_context=_context()
        )

        assert top[0]["reason_source"] == "sub_sector_attribution"

    def test_falls_back_when_generate_text_returns_none(self):
        top = self._top()

        SectorDriverService(analyzer=_analyzer(None)).annotate(
            top, [], news=_news(), catalyst_context=_context()
        )

        assert top[0]["reason_source"] == "sub_sector_attribution"

    def test_handles_empty_and_malformed_inputs(self):
        empty = {"total": 0, "llm": 0, "attribution": 0, "empty": 0}
        assert SectorDriverService().annotate(None, None) == empty
        assert SectorDriverService().annotate([{"name": ""}, "bad"], []) == empty


class TestMarketAnalyzerIntegration:
    def _ma(self):
        from src.market_analyzer import MarketAnalyzer

        ma = MarketAnalyzer.__new__(MarketAnalyzer)
        ma.config = SimpleNamespace(market_review_color_scheme="green_up", report_language="zh")
        ma.region = "cn"
        return ma

    def test_table_is_top3_with_separate_gain_loss_headers(self):
        from src.market_analyzer import MarketOverview

        ma = self._ma()
        overview = MarketOverview(
            date="2026-09-24",
            top_sectors=[
                {"name": "煤炭", "change_pct": 1.04, "reason": "发改委保供表态"},
                {"name": "银行", "change_pct": 0.68, "reason": "高股息防御"},
                {"name": "纺织服饰", "change_pct": 0.52, "reason": "纺织制造走强"},
                {"name": "多余行业", "change_pct": 0.40, "reason": "不应出现"},
            ],
            bottom_sectors=[{"name": "有色金属", "change_pct": -2.79}],
        )

        block = ma._build_sector_block(overview)

        assert "#### 领涨行业 Top 3" in block
        assert "| 行业 | 涨幅 | 异动原因 |" in block
        assert "#### 领跌行业 Top 3" in block
        assert "| 行业 | 跌幅 | 异动原因 |" in block
        assert "| 煤炭 | +1.04% | 发改委保供表态 |" in block
        assert "| 有色金属 | -2.79% | - |" in block
        # 只保留 Top3，且不再有排名列
        assert "多余行业" not in block
        assert "| 排名 |" not in block

    def test_search_sector_news_queries_each_industry(self):
        ma = self._ma()
        ma.config = SimpleNamespace(report_language="zh")
        calls = []

        def search_stock_news(stock_code, stock_name, max_results, focus_keywords):
            calls.append({"name": stock_name, "keywords": focus_keywords})
            return SimpleNamespace(results=[{"title": f"{stock_name}新闻"}])

        ma.search_service = SimpleNamespace(search_stock_news=search_stock_news)

        result = ma._search_sector_news([{"name": "煤炭"}, {"name": "银行"}, {"name": "煤炭"}])

        assert set(result) == {"煤炭", "银行"}
        assert [c["name"] for c in calls] == ["煤炭行业", "银行行业"]  # 去重后逐个检索
        assert "煤炭" in calls[0]["keywords"]

    def test_search_sector_news_can_be_disabled(self):
        ma = self._ma()
        ma.config = SimpleNamespace(market_sector_news_search_enabled=False)
        ma.search_service = SimpleNamespace(
            search_stock_news=lambda **kw: (_ for _ in ()).throw(AssertionError("不该被调用"))
        )

        assert ma._search_sector_news([{"name": "煤炭"}]) == {}

    def test_search_sector_news_isolates_per_industry_failure(self):
        ma = self._ma()
        ma.config = SimpleNamespace(report_language="zh")

        def search_stock_news(stock_code, stock_name, max_results, focus_keywords):
            if stock_name.startswith("煤炭"):
                raise RuntimeError("search down")
            return SimpleNamespace(results=[{"title": "ok"}])

        ma.search_service = SimpleNamespace(search_stock_news=search_stock_news)

        result = ma._search_sector_news([{"name": "煤炭"}, {"name": "银行"}])

        assert "煤炭" not in result
        assert "银行" in result

    def test_annotate_never_raises_and_still_attributes(self):
        from src.market_analyzer import MarketOverview
        from src.core.market_profile import get_profile

        ma = self._ma()
        ma.profile = get_profile("cn")
        ma.search_service = None
        ma.analyzer = SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        ma.data_manager = SimpleNamespace(
            get_sector_catalyst_context=lambda sectors: _context()
        )
        overview = MarketOverview(
            date="2026-09-24",
            top_sectors=[{"name": "煤炭", "change_pct": 1.04}],
            bottom_sectors=[],
        )

        ma._annotate_sector_reasons(overview, news=_news())

        assert "焦炭Ⅱ" in overview.top_sectors[0]["reason"]

    def test_annotate_survives_catalyst_source_failure(self):
        from src.market_analyzer import MarketOverview
        from src.core.market_profile import get_profile

        ma = self._ma()
        ma.profile = get_profile("cn")
        ma.search_service = None
        ma.analyzer = None
        ma.data_manager = SimpleNamespace(
            get_sector_catalyst_context=lambda sectors: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        overview = MarketOverview(
            date="2026-09-24",
            top_sectors=[{"name": "煤炭", "change_pct": 1.04}],
            bottom_sectors=[],
        )

        ma._annotate_sector_reasons(overview, news=_news())

        assert "reason" not in overview.top_sectors[0]


class TestNewsSourceDegradation:
    """搜索 provider 未配置是常见部署形态，此时必须自动改用 akshare 板块新闻。"""

    def _ma(self):
        from src.market_analyzer import MarketAnalyzer

        ma = MarketAnalyzer.__new__(MarketAnalyzer)
        ma.config = SimpleNamespace(report_language="zh")
        ma.region = "cn"
        return ma

    def test_uses_akshare_when_no_search_service(self):
        ma = self._ma()
        ma.search_service = None
        ma.data_manager = SimpleNamespace(
            get_sector_news=lambda names, max_items=3: {n: [{"title": f"{n}板块快讯"}] for n in names}
        )

        result = ma._search_sector_news([{"name": "煤炭"}, {"name": "银行"}])

        assert result["煤炭"][0]["title"] == "煤炭板块快讯"
        assert result["银行"][0]["title"] == "银行板块快讯"

    def test_akshare_only_fills_industries_web_search_missed(self):
        ma = self._ma()

        def search_stock_news(stock_code, stock_name, max_results, focus_keywords):
            if stock_name.startswith("煤炭"):
                return SimpleNamespace(results=[{"title": "网页搜索命中"}])
            return SimpleNamespace(results=[])

        ma.search_service = SimpleNamespace(search_stock_news=search_stock_news)
        asked = []

        def get_sector_news(names, max_items=3):
            asked.extend(names)
            return {n: [{"title": f"{n}兜底"}] for n in names}

        ma.data_manager = SimpleNamespace(get_sector_news=get_sector_news)

        result = ma._search_sector_news([{"name": "煤炭"}, {"name": "银行"}])

        assert result["煤炭"][0]["title"] == "网页搜索命中"   # 有 key 时以网页搜索为优
        assert asked == ["银行"]                              # 只为未命中的行业兜底
        assert result["银行"][0]["title"] == "银行兜底"

    def test_disabled_switch_skips_both_sources(self):
        ma = self._ma()
        ma.config = SimpleNamespace(market_sector_news_search_enabled=False)
        ma.search_service = SimpleNamespace(
            search_stock_news=lambda **kw: (_ for _ in ()).throw(AssertionError("不该被调用"))
        )
        ma.data_manager = SimpleNamespace(
            get_sector_news=lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该被调用"))
        )

        assert ma._search_sector_news([{"name": "煤炭"}]) == {}

    def test_akshare_failure_is_isolated(self):
        ma = self._ma()
        ma.search_service = None
        ma.data_manager = SimpleNamespace(
            get_sector_news=lambda names, max_items=3: (_ for _ in ()).throw(RuntimeError("down"))
        )

        assert ma._search_sector_news([{"name": "煤炭"}]) == {}

    def test_market_wire_news_backfills_empty_market_news(self):
        from src.market_analyzer import MarketOverview
        from src.core.market_profile import get_profile

        ma = self._ma()
        ma.config = SimpleNamespace(report_language="zh", market_sector_reason_enabled=True)
        ma.profile = get_profile("cn")
        ma.search_service = None
        captured = {}

        def gen(prompt, max_tokens=None, temperature=None):
            captured["prompt"] = prompt
            return "煤炭 => 动力煤价格上涨带动"

        ma.analyzer = SimpleNamespace(is_available=lambda: True, generate_text=gen)
        ma.data_manager = SimpleNamespace(
            get_sector_catalyst_context=lambda sectors: {},
            get_sector_news=lambda names, max_items=3: {},
            get_market_wire_news=lambda limit: [{"title": "财联社电报：动力煤价格走高"}],
        )
        overview = MarketOverview(
            date="2026-09-24",
            top_sectors=[{"name": "煤炭", "change_pct": 0.59}],
            bottom_sectors=[],
        )

        ma._annotate_sector_reasons(overview, news=[])

        assert "财联社电报：动力煤价格走高" in captured["prompt"]
        assert overview.top_sectors[0]["reason"] == "动力煤价格上涨带动"

    def test_wire_news_failure_is_isolated(self):
        ma = self._ma()
        ma.data_manager = SimpleNamespace(
            get_market_wire_news=lambda limit: (_ for _ in ()).throw(RuntimeError("down"))
        )

        assert ma._get_market_wire_news() == []
