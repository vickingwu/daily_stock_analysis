# -*- coding: utf-8 -*-
"""Tests for the sector driver (板块异动原因) enrichment.

Covers:
- 事实层：仅靠板块榜单自带字段（涨跌家数、领涨个股）生成原因，中英双语
- 解读层：LLM 返回「板块名 => 原因」的解析、噪声容忍、无依据与歧义时的丢弃
- 降级路径：未配置模型 / 开关关闭 / 无新闻 / 调用抛错时退回事实层
- 表格渲染：异动原因列的占位、管道符转义与长度截断
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
        generate_text.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        if raises is not None:
            raise raises
        return response

    generate_text.calls = []
    return SimpleNamespace(generate_text=generate_text, is_available=lambda: available)


def _news():
    return [
        {
            "title": "发改委强调迎峰度夏煤炭保供",
            "snippet": "要求主产区稳产增产，动力煤长协价格机制优化",
            "source": "新华社",
        },
        {
            "title": "多家云厂商上调AI算力资本开支指引",
            "snippet": "光模块与服务器订单能见度延长至明年",
            "source": "证券时报",
        },
    ]


class TestFactualReason:
    """事实层：不依赖任何网络请求与模型"""

    def test_uses_breadth_and_leader_for_leading_sector(self):
        top = [{
            "name": "煤炭行业",
            "change_pct": 5.22,
            "up_count": 35,
            "down_count": 0,
            "leader_stock": "云煤能源",
            "leader_change_pct": 10.12,
        }]

        stats = SectorDriverService().annotate(top, [])

        assert top[0]["reason"] == "35涨0跌，板块普涨，龙头云煤能源 +10.12%"
        assert top[0]["reason_source"] == "board_internals"
        assert stats == {"total": 1, "llm": 0, "factual": 1, "empty": 0}

    def test_lagging_sector_avoids_calling_best_performer_a_leader(self):
        bottom = [{
            "name": "银行",
            "change_pct": -1.30,
            "up_count": 2,
            "down_count": 38,
            "leader_stock": "成都银行",
            "leader_change_pct": 0.45,
        }]

        SectorDriverService().annotate([], bottom)

        assert bottom[0]["reason"] == "2涨38跌，板块普跌，板块内最强成都银行 +0.45%"

    def test_marks_mixed_breadth_when_neither_side_dominates(self):
        top = [{"name": "软件开发", "change_pct": 1.1, "up_count": 90, "down_count": 89}]

        SectorDriverService().annotate(top, [])

        assert top[0]["reason"] == "90涨89跌，板块内分化"

    def test_falls_back_to_leader_only_when_breadth_missing(self):
        """新浪备用源没有涨跌家数，只能给出领涨个股。"""
        top = [{"name": "煤炭", "change_pct": 4.0, "leader_stock": "云煤能源", "leader_change_pct": 9.98}]

        SectorDriverService().annotate(top, [])

        assert top[0]["reason"] == "龙头云煤能源 +9.98%"

    def test_leaves_reason_absent_when_no_internals_available(self):
        top = [{"name": "煤炭行业", "change_pct": 5.22}]

        stats = SectorDriverService().annotate(top, [])

        assert "reason" not in top[0]
        assert stats == {"total": 1, "llm": 0, "factual": 0, "empty": 1}

    def test_english_language_renders_english_facts(self):
        top = [{
            "name": "Coal",
            "change_pct": 5.22,
            "up_count": 35,
            "down_count": 0,
            "leader_stock": "Yunmei Energy",
            "leader_change_pct": 10.12,
        }]

        SectorDriverService().annotate(top, [], language="en")

        assert top[0]["reason"] == "35 up / 0 down, broad advance; led by Yunmei Energy +10.12%"


class TestLlmReason:
    def test_parses_llm_lines_and_prefers_them_over_facts(self):
        analyzer = _analyzer(
            "煤炭行业 => 发改委保供增产，长协价机制优化\n"
            "光模块 => 云厂商上调AI资本开支\n"
        )
        top = [
            {"name": "煤炭行业", "change_pct": 5.22, "up_count": 35, "down_count": 0},
            {"name": "光模块", "change_pct": 4.1},
        ]

        stats = SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert top[0]["reason"] == "发改委保供增产，长协价机制优化"
        assert top[0]["reason_source"] == "llm"
        assert top[1]["reason"] == "云厂商上调AI资本开支"
        assert stats == {"total": 2, "llm": 2, "factual": 0, "empty": 0}

    def test_prompt_carries_sector_clues_and_news(self):
        analyzer = _analyzer("煤炭行业 => 保供预期升温")
        top = [{"name": "煤炭行业", "change_pct": 5.22, "up_count": 35, "down_count": 0}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        prompt = analyzer.generate_text.calls[0]["prompt"]
        assert "领涨 煤炭行业 +5.22%" in prompt
        assert "35涨0跌，板块普涨" in prompt
        assert "发改委强调迎峰度夏煤炭保供" in prompt
        assert "不得编造" in prompt

    def test_tolerates_bullets_numbering_and_trailing_change_pct(self):
        analyzer = _analyzer(
            "```\n"
            "1. 煤炭行业 (+5.22%)：发改委保供增产\n"
            "- 光模块 → 算力订单能见度延长\n"
            "```"
        )
        top = [{"name": "煤炭行业", "change_pct": 5.22}, {"name": "光模块", "change_pct": 4.1}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert top[0]["reason"] == "发改委保供增产"
        assert top[1]["reason"] == "算力订单能见度延长"

    @pytest.mark.parametrize("token", ["无", "none", "N/A", "暂无数据", "-"])
    def test_drops_reasons_the_model_marked_unavailable(self, token):
        analyzer = _analyzer(f"煤炭行业 => {token}")
        top = [{"name": "煤炭行业", "change_pct": 5.22, "up_count": 35, "down_count": 0}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        # 退回事实层而不是写入 "无"
        assert top[0]["reason"] == "35涨0跌，板块普涨"
        assert top[0]["reason_source"] == "board_internals"

    def test_drops_ambiguous_sector_name_instead_of_guessing(self):
        analyzer = _analyzer("煤炭 => 保供预期升温")
        top = [{"name": "煤炭行业", "change_pct": 5.22}, {"name": "煤炭开采", "change_pct": 4.9}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert "reason" not in top[0]
        assert "reason" not in top[1]

    def test_ignores_unknown_sector_names(self):
        analyzer = _analyzer("白酒 => 春节备货超预期")
        top = [{"name": "煤炭行业", "change_pct": 5.22}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert "reason" not in top[0]

    def test_truncates_overlong_llm_reason(self):
        analyzer = _analyzer(f"煤炭行业 => {'政策预期' * 20}")
        top = [{"name": "煤炭行业", "change_pct": 5.22}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert len(top[0]["reason"]) == SectorDriverService.MAX_REASON_CHARS
        assert top[0]["reason"].endswith("...")

    def test_strips_pipe_to_keep_markdown_table_intact(self):
        analyzer = _analyzer("煤炭行业 => 保供增产 | 长协优化")
        top = [{"name": "煤炭行业", "change_pct": 5.22}]

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert "|" not in top[0]["reason"]
        assert top[0]["reason"] == "保供增产 / 长协优化"


class TestLlmDegradation:
    def _top(self):
        return [{"name": "煤炭行业", "change_pct": 5.22, "up_count": 35, "down_count": 0}]

    def test_skips_llm_without_news(self):
        analyzer = _analyzer("煤炭行业 => 不该被调用")
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=[])

        assert analyzer.generate_text.calls == []
        assert top[0]["reason_source"] == "board_internals"

    def test_skips_llm_when_config_disables_it(self):
        analyzer = _analyzer("煤炭行业 => 不该被调用")
        config = SimpleNamespace(market_sector_reason_enabled=False)
        top = self._top()

        SectorDriverService(config=config, analyzer=analyzer).annotate(top, [], news=_news())

        assert analyzer.generate_text.calls == []
        assert top[0]["reason_source"] == "board_internals"

    def test_skips_llm_when_analyzer_unavailable(self):
        analyzer = _analyzer("煤炭行业 => 不该被调用", available=False)
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert analyzer.generate_text.calls == []
        assert top[0]["reason_source"] == "board_internals"

    def test_falls_back_when_generate_text_raises(self):
        analyzer = _analyzer(raises=RuntimeError("llm down"))
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert top[0]["reason"] == "35涨0跌，板块普涨"
        assert top[0]["reason_source"] == "board_internals"

    def test_falls_back_when_generate_text_returns_none(self):
        analyzer = _analyzer(None)
        top = self._top()

        SectorDriverService(analyzer=analyzer).annotate(top, [], news=_news())

        assert top[0]["reason_source"] == "board_internals"

    def test_handles_empty_and_malformed_sector_lists(self):
        assert SectorDriverService().annotate(None, None) == {
            "total": 0, "llm": 0, "factual": 0, "empty": 0,
        }
        assert SectorDriverService().annotate([{"name": ""}, "bad"], []) == {
            "total": 0, "llm": 0, "factual": 0, "empty": 0,
        }


class TestMarketAnalyzerIntegration:
    def _analyzer_instance(self):
        from src.market_analyzer import MarketAnalyzer

        return MarketAnalyzer.__new__(MarketAnalyzer)

    def test_sector_block_renders_driver_column(self):
        from src.market_analyzer import MarketOverview

        ma = self._analyzer_instance()
        ma.config = SimpleNamespace(market_review_color_scheme="green_up", report_language="zh")
        overview = MarketOverview(
            date="2026-03-05",
            top_sectors=[{"name": "煤炭行业", "change_pct": 5.22, "reason": "发改委保供增产"}],
            bottom_sectors=[{"name": "银行", "change_pct": -1.3}],
        )

        block = ma._build_sector_block(overview)

        assert "| 排名 | 板块 | 涨跌幅 | 异动原因 |" in block
        assert "| 1 | 煤炭行业 | +5.22% | 发改委保供增产 |" in block
        assert "| 1 | 银行 | -1.30% | - |" in block

    def test_prompt_sectors_include_reason_hint(self):
        ma = self._analyzer_instance()

        text = ma._format_prompt_sectors([
            {"name": "煤炭行业", "change_pct": 5.22, "reason": "发改委保供增产"},
            {"name": "光模块", "change_pct": 4.1},
        ])

        assert text == "煤炭行业(+5.22%)[发改委保供增产], 光模块(+4.10%)"

    def test_annotate_sector_reasons_never_raises(self):
        from src.market_analyzer import MarketOverview
        from src.core.market_profile import get_profile

        ma = self._analyzer_instance()
        ma.config = SimpleNamespace(report_language="zh")
        ma.region = "cn"
        ma.profile = get_profile("cn")
        ma.analyzer = SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        overview = MarketOverview(
            date="2026-03-05",
            top_sectors=[{"name": "煤炭行业", "change_pct": 5.22, "up_count": 35, "down_count": 0}],
            bottom_sectors=[],
        )

        ma._annotate_sector_reasons(overview, news=_news())

        # analyzer 探测失败也要降级到事实层，而不是冒泡到复盘主流程
        assert overview.top_sectors[0]["reason"] == "35涨0跌，板块普涨"
