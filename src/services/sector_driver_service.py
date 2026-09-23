# -*- coding: utf-8 -*-
"""板块异动原因服务。

为大盘复盘的领涨 / 领跌板块各补一条一句话「异动原因」，供推送表格展示。

原因来源分两层，由高到低：

1. 解读层（可选）：把当日已抓取的市场新闻与板块线索一起交给大模型，归纳板块
   「为什么动」。未配置模型、没有可用新闻、调用失败或模型给不出依据时自动降级。
2. 事实层：只使用板块榜单自带的板块内部结构（上涨 / 下跌家数、领涨个股），
   不额外请求任何接口，始终可用。

两层都拿不到时留空，表格显示 "-"。本模块不向上抛异常，保证单点失败不影响
大盘复盘主流程。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 模型表示「给不出原因」时可能返回的各种写法，统一视为无结果
_REASON_UNAVAILABLE_TOKENS = frozenset({
    "-", "--", "—", "无", "无。", "暂无", "暂无数据", "暂无依据", "未知", "不详",
    "none", "n/a", "na", "null", "unknown", "unclear", "no reason", "no data",
})

# 解析「板块名 => 原因」，容忍模型附带序号、项目符号与多种分隔符
_REASON_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*•]|\d+\s*[.)、])?\s*(?P<name>.+?)\s*(?:=>|=＞|->|→|::|：|:)\s*(?P<reason>.+?)\s*$"
)

_MARKDOWN_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _text(value: Any) -> str:
    """折叠空白后的纯文本，None 安全。"""
    return " ".join(str(value if value is not None else "").split())


def _shorten(value: str, limit: int) -> str:
    text = _text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip("，,；;：: ") + "..."


def _news_field(item: Any, field: str) -> str:
    """读取新闻字段，兼容 SearchResult 对象与 dict 两种形态。"""
    if hasattr(item, field):
        return _text(getattr(item, field, ""))
    if isinstance(item, dict):
        return _text(item.get(field, ""))
    return ""


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class SectorDriverService:
    """推断板块异动原因，并就地写回板块字典的 ``reason`` 字段。"""

    # 原因文案长度上限：中文按字符数控制，英文放宽以容纳单词
    MAX_REASON_CHARS = 30
    MAX_REASON_CHARS_EN = 64

    _LLM_MAX_TOKENS = 700
    _LLM_TEMPERATURE = 0.2
    _MAX_NEWS_ITEMS = 8
    _NEWS_TITLE_LIMIT = 80
    _NEWS_SNIPPET_LIMIT = 110

    def __init__(self, config: Any = None, analyzer: Any = None) -> None:
        self.config = config
        self.analyzer = analyzer

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def annotate(
        self,
        top_sectors: Optional[List[Dict[str, Any]]],
        bottom_sectors: Optional[List[Dict[str, Any]]],
        *,
        news: Optional[Sequence[Any]] = None,
        language: str = "zh",
    ) -> Dict[str, int]:
        """给领涨 / 领跌板块写入 ``reason`` 与 ``reason_source``。

        Args:
            top_sectors: 领涨板块列表，元素为 dict，就地修改
            bottom_sectors: 领跌板块列表，元素为 dict，就地修改
            news: 当日市场新闻（SearchResult 或 dict 列表），用于解读层
            language: 报告语言，"en" 输出英文原因，其余按中文处理

        Returns:
            Dict: 各来源命中数量，供调用方打日志，形如
            ``{"total": 10, "llm": 6, "factual": 3, "empty": 1}``
        """
        entries = self._collect_entries(top_sectors, bottom_sectors)
        stats = {"total": len(entries), "llm": 0, "factual": 0, "empty": 0}
        if not entries:
            return stats

        llm_reasons = self._build_llm_reasons(entries, news=news, language=language)

        for sector, direction in entries:
            name = _text(sector.get("name"))
            reason = self._clean_reason(llm_reasons.get(name), language)
            source = "llm" if reason else ""
            if not reason:
                reason = self._clean_reason(
                    self._build_factual_reason(sector, direction, language), language
                )
                source = "board_internals" if reason else ""

            if not reason:
                stats["empty"] += 1
                continue

            sector["reason"] = reason
            sector["reason_source"] = source
            stats["llm" if source == "llm" else "factual"] += 1

        return stats

    # ------------------------------------------------------------------
    # 事实层：板块内部结构
    # ------------------------------------------------------------------
    def _build_factual_reason(
        self,
        sector: Dict[str, Any],
        direction: str,
        language: str,
    ) -> str:
        """仅用板块榜单自带字段描述异动，无需额外请求。"""
        parts: List[str] = []

        breadth = self._describe_breadth(
            _to_int(sector.get("up_count")),
            _to_int(sector.get("down_count")),
            language,
        )
        if breadth:
            parts.append(breadth)

        leader = self._describe_leader(
            _text(sector.get("leader_stock")),
            _to_float(sector.get("leader_change_pct")),
            direction,
            language,
        )
        if leader:
            parts.append(leader)

        return ("; " if language == "en" else "，").join(parts)

    @staticmethod
    def _describe_breadth(
        up_count: Optional[int],
        down_count: Optional[int],
        language: str,
    ) -> str:
        if up_count is None or down_count is None:
            return ""
        total = up_count + down_count
        if total <= 0:
            return ""
        up_ratio = up_count / total
        if language == "en":
            if up_ratio >= 0.8:
                tone = "broad advance"
            elif up_ratio <= 0.2:
                tone = "broad decline"
            else:
                tone = "mixed"
            return f"{up_count} up / {down_count} down, {tone}"
        if up_ratio >= 0.8:
            tone = "板块普涨"
        elif up_ratio <= 0.2:
            tone = "板块普跌"
        else:
            tone = "板块内分化"
        return f"{up_count}涨{down_count}跌，{tone}"

    @staticmethod
    def _describe_leader(
        leader: str,
        leader_change_pct: Optional[float],
        direction: str,
        language: str,
    ) -> str:
        if not leader:
            return ""
        pct_text = "" if leader_change_pct is None else f" {leader_change_pct:+.2f}%"
        if language == "en":
            label = "led by" if direction == "up" else "best performer"
            return f"{label} {leader}{pct_text}"
        # 领跌榜里「领涨股票」只是板块内最强个股，措辞需区分，避免误读为板块上涨
        label = "龙头" if direction == "up" else "板块内最强"
        return f"{label}{leader}{pct_text}"

    # ------------------------------------------------------------------
    # 解读层：LLM
    # ------------------------------------------------------------------
    def _build_llm_reasons(
        self,
        entries: List[Tuple[Dict[str, Any], str]],
        *,
        news: Optional[Sequence[Any]],
        language: str,
    ) -> Dict[str, str]:
        """返回 ``{板块名: 原因}``；不可用时返回空字典。"""
        if not self._llm_enabled():
            return {}

        news_text = self._format_news(news, language)
        if not news_text:
            # 没有新闻就没有「为什么动」的依据，直接交给事实层，避免模型凭空编造
            logger.info("[板块异动] action=llm_reason status=skipped reason=no_news")
            return {}

        prompt = self._build_prompt(entries, news_text, language)
        try:
            response = self.analyzer.generate_text(
                prompt,
                max_tokens=self._LLM_MAX_TOKENS,
                temperature=self._LLM_TEMPERATURE,
            )
        except Exception as exc:  # pragma: no cover - generate_text 内部已兜底
            logger.warning("[板块异动] action=llm_reason status=failed error=%s", exc)
            return {}

        if not response:
            logger.info("[板块异动] action=llm_reason status=empty_response")
            return {}

        reasons = self._parse_reasons(response, [_text(s.get("name")) for s, _ in entries])
        logger.info(
            "[板块异动] action=llm_reason status=success parsed=%d/%d",
            len(reasons),
            len(entries),
        )
        return reasons

    def _llm_enabled(self) -> bool:
        if getattr(self.config, "market_sector_reason_enabled", True) is not True:
            return False
        if self.analyzer is None:
            return False
        try:
            return bool(self.analyzer.is_available())
        except Exception:
            return False

    def _build_prompt(
        self,
        entries: List[Tuple[Dict[str, Any], str]],
        news_text: str,
        language: str,
    ) -> str:
        clues = "\n".join(self._format_clue(sector, direction, language) for sector, direction in entries)
        if language == "en":
            return (
                "You are an equity market recap assistant. For each sector below, write one "
                "line explaining why it led or lagged today.\n\n"
                "Output format (strict):\n"
                "- One sector per line, exactly: Sector => Reason\n"
                "- Output only those lines. No tables, numbering, headings or commentary.\n\n"
                "Content rules:\n"
                f"- Keep each reason under {self.MAX_REASON_CHARS_EN} characters and name the driver "
                "(policy, news, event, fund flow, leading stock). Do not restate the percentage change.\n"
                "- Use only the sector clues and news below. Never invent events, companies, "
                "policies or figures.\n"
                "- If a sector has no supporting evidence, write: Sector => none\n\n"
                f"[Sector clues]\n{clues}\n\n"
                f"[Market news]\n{news_text}\n"
            )
        return (
            "你是 A 股盘后复盘助手。请为下面每个板块写一条「异动原因」，解释它今天为什么领涨或领跌。\n\n"
            "输出格式（严格遵守）：\n"
            "- 每行一个板块，格式为：板块名 => 原因\n"
            "- 只输出这些行，不要表格、序号、标题或额外说明\n\n"
            "内容要求：\n"
            f"- 每条原因不超过 {self.MAX_REASON_CHARS} 个字，写驱动因素（政策、消息、事件、资金、龙头股），"
            "不要重复涨跌幅数字\n"
            "- 只能依据下面的【板块线索】和【市场新闻】，不得编造事件、公司、政策或数据\n"
            "- 找不到依据的板块，原因写「无」\n\n"
            f"【板块线索】\n{clues}\n\n"
            f"【市场新闻】\n{news_text}\n"
        )

    def _format_clue(self, sector: Dict[str, Any], direction: str, language: str) -> str:
        name = _text(sector.get("name")) or "-"
        change_pct = _to_float(sector.get("change_pct"))
        change_text = "N/A" if change_pct is None else f"{change_pct:+.2f}%"
        if language == "en":
            label = "Leading" if direction == "up" else "Lagging"
        else:
            label = "领涨" if direction == "up" else "领跌"
        segments = [f"{label} {name} {change_text}"]

        factual = self._build_factual_reason(sector, direction, language)
        if factual:
            segments.append(factual)
        return ("; " if language == "en" else "；").join(segments)

    def _format_news(self, news: Optional[Sequence[Any]], language: str) -> str:
        lines: List[str] = []
        for item in list(news or [])[: self._MAX_NEWS_ITEMS]:
            title = _shorten(_news_field(item, "title"), self._NEWS_TITLE_LIMIT)
            if not title:
                continue
            source = _shorten(_news_field(item, "source"), 30)
            snippet = _shorten(_news_field(item, "snippet"), self._NEWS_SNIPPET_LIMIT)
            head = f"{len(lines) + 1}. {title}"
            if source:
                head += f" ({source})" if language == "en" else f"（{source}）"
            lines.append(f"{head}\n   {snippet}" if snippet else head)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 解析与清洗
    # ------------------------------------------------------------------
    @classmethod
    def _parse_reasons(cls, response: str, known_names: Sequence[str]) -> Dict[str, str]:
        known = [name for name in known_names if name]
        reasons: Dict[str, str] = {}
        for raw_line in str(response or "").splitlines():
            line = _MARKDOWN_FENCE_PATTERN.sub("", raw_line).strip()
            if not line:
                continue
            match = _REASON_LINE_PATTERN.match(line)
            if not match:
                continue
            name = cls._match_sector_name(match.group("name"), known)
            if not name or name in reasons:
                continue
            reason = _text(match.group("reason")).strip("「」\"'` 　")
            if not reason or reason.lower() in _REASON_UNAVAILABLE_TOKENS:
                continue
            reasons[name] = reason
        return reasons

    @staticmethod
    def _match_sector_name(raw_name: str, known_names: Sequence[str]) -> str:
        """把模型给出的板块名对回榜单里的板块名，歧义时放弃以免错配。"""
        candidate = _text(raw_name)
        if not candidate:
            return ""
        if candidate in known_names:
            return candidate
        matches = {
            name for name in known_names
            if name and (name in candidate or candidate in name)
        }
        return matches.pop() if len(matches) == 1 else ""

    @classmethod
    def _clean_reason(cls, value: Any, language: str) -> str:
        """压成单行、去掉会破坏 Markdown 表格的字符，并限制长度。"""
        text = _text(value).replace("|", "/").strip("「」\"'` 　")
        if not text or text.lower() in _REASON_UNAVAILABLE_TOKENS:
            return ""
        limit = cls.MAX_REASON_CHARS_EN if language == "en" else cls.MAX_REASON_CHARS
        return _shorten(text, limit)

    @staticmethod
    def _collect_entries(
        top_sectors: Optional[List[Dict[str, Any]]],
        bottom_sectors: Optional[List[Dict[str, Any]]],
    ) -> List[Tuple[Dict[str, Any], str]]:
        entries: List[Tuple[Dict[str, Any], str]] = []
        for sectors, direction in ((top_sectors, "up"), (bottom_sectors, "down")):
            for sector in sectors or []:
                if isinstance(sector, dict) and _text(sector.get("name")):
                    entries.append((sector, direction))
        return entries
