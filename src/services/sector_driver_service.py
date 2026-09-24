# -*- coding: utf-8 -*-
"""行业异动原因服务。

为大盘复盘的领涨 / 领跌行业各补一条「异动原因」，说明该行业当天为什么动，
供推送表格展示。目标是事件催化口径（政策、消息、外盘、资金、龙头股），
而不是把涨跌幅换个说法复述一遍。

原因来源分两层，由高到低：

1. 解读层：把当日检索到的行业新闻，与行业结构材料（申万二级子板块涨跌、
   板块内涨停股、领涨领跌个股）一起交给大模型，归纳成一句催化说明。
   模型只能依据给定材料，给不出依据的行业必须弃权。
2. 归因层：模型弃权或不可用时，只做子板块归因（「主要由玻璃玻纤拖累」），
   这与券商复盘里「主要是航运港口板块走弱」同类，属于对「为什么」的回答；
   不使用涨跌家数、涨跌幅复述这类与原因无关的描述。

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
    "-", "--", "—", "无", "无。", "暂无", "暂无数据", "暂无依据", "未知", "不详", "弃权",
    "none", "n/a", "na", "null", "unknown", "unclear", "no reason", "no data", "skip",
})

# 解析「行业名 => 原因」，容忍模型附带序号、项目符号与多种分隔符
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


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: Any) -> str:
    num = _to_float(value)
    return "N/A" if num is None else f"{num:+.2f}%"


class SectorDriverService:
    """推断行业异动原因，并就地写回行业字典的 ``reason`` 字段。"""

    # 原因文案长度上限：对齐券商复盘口径，需要容纳政策名、个股名与因果链
    MAX_REASON_CHARS = 120
    MAX_REASON_CHARS_EN = 240

    _LLM_MAX_TOKENS = 3000
    _LLM_TEMPERATURE = 0.3
    _MAX_MARKET_NEWS = 6
    _MAX_SECTOR_NEWS = 4
    _NEWS_TITLE_LIMIT = 90
    _NEWS_SNIPPET_LIMIT = 130
    _MAX_SUB_SECTORS = 5
    _MAX_STOCK_EXAMPLES = 4

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
        sector_news: Optional[Dict[str, Sequence[Any]]] = None,
        catalyst_context: Optional[Dict[str, Dict[str, Any]]] = None,
        language: str = "zh",
    ) -> Dict[str, int]:
        """给领涨 / 领跌行业写入 ``reason`` 与 ``reason_source``。

        Args:
            top_sectors: 领涨行业列表，元素为 dict，就地修改
            bottom_sectors: 领跌行业列表，元素为 dict，就地修改
            news: 市场级新闻（SearchResult 或 dict 列表）
            sector_news: ``{行业名: 该行业新闻列表}``，按行业单独检索的结果
            catalyst_context: ``{行业名: {sub_sectors, limit_up_stocks,
                top_gainers, top_losers}}``，见 AkshareFetcher.get_sector_catalyst_context
            language: 报告语言，"en" 输出英文原因，其余按中文处理

        Returns:
            Dict: 各来源命中数量，供调用方打日志，形如
            ``{"total": 6, "llm": 5, "attribution": 1, "empty": 0}``
        """
        entries = self._collect_entries(top_sectors, bottom_sectors)
        stats = {"total": len(entries), "llm": 0, "attribution": 0, "empty": 0}
        if not entries:
            return stats

        context = catalyst_context or {}
        per_sector_news = sector_news or {}

        llm_reasons = self._build_llm_reasons(
            entries,
            news=news,
            sector_news=per_sector_news,
            catalyst_context=context,
            language=language,
        )

        for sector, direction in entries:
            name = _text(sector.get("name"))
            reason = self._clean_reason(llm_reasons.get(name), language)
            source = "llm" if reason else ""
            if not reason:
                reason = self._clean_reason(
                    self._build_attribution_reason(
                        context.get(name) or {}, direction, language
                    ),
                    language,
                )
                source = "sub_sector_attribution" if reason else ""

            if not reason:
                stats["empty"] += 1
                continue

            sector["reason"] = reason
            sector["reason_source"] = source
            stats["llm" if source == "llm" else "attribution"] += 1

        return stats

    # ------------------------------------------------------------------
    # 归因层：子板块 / 涨停股（不含涨跌幅复述）
    # ------------------------------------------------------------------
    def _build_attribution_reason(
        self,
        material: Dict[str, Any],
        direction: str,
        language: str,
    ) -> str:
        """模型弃权时的兜底：指出行业内部是哪一块在带动，而不是复述涨跌幅。"""
        parts: List[str] = []

        subs = [s for s in (material.get("sub_sectors") or []) if isinstance(s, dict)]
        if subs:
            picked = subs[0] if direction == "up" else subs[-1]
            sub_name = _text(picked.get("name"))
            if sub_name:
                if language == "en":
                    verb = "led by" if direction == "up" else "dragged down by"
                    parts.append(f"mainly {verb} {sub_name} ({_fmt_pct(picked.get('change_pct'))})")
                else:
                    verb = "带动" if direction == "up" else "拖累"
                    parts.append(f"主要由{sub_name}({_fmt_pct(picked.get('change_pct'))}){verb}")

        limit_ups = [s for s in (material.get("limit_up_stocks") or []) if isinstance(s, dict)]
        names = [_text(s.get("name")) for s in limit_ups if _text(s.get("name"))]
        if names and direction == "up":
            listed = "、".join(names[:3]) if language != "en" else ", ".join(names[:3])
            if language == "en":
                parts.append(f"{listed} hit the daily limit")
            else:
                parts.append(f"{listed}等涨停")

        if not parts:
            return ""

        joined = ("; " if language == "en" else "，").join(parts)
        if language == "en":
            return f"{joined}; no clear news catalyst found"
        return f"{joined}；未检索到明确消息催化"

    # ------------------------------------------------------------------
    # 解读层：LLM
    # ------------------------------------------------------------------
    def _build_llm_reasons(
        self,
        entries: List[Tuple[Dict[str, Any], str]],
        *,
        news: Optional[Sequence[Any]],
        sector_news: Dict[str, Sequence[Any]],
        catalyst_context: Dict[str, Dict[str, Any]],
        language: str,
    ) -> Dict[str, str]:
        """返回 ``{行业名: 原因}``；不可用时返回空字典。"""
        if not self._llm_enabled():
            return {}

        has_sector_news = any(sector_news.get(_text(s.get("name"))) for s, _ in entries)
        market_news_text = self._format_news(news, language, limit=self._MAX_MARKET_NEWS)
        if not has_sector_news and not market_news_text:
            # 没有任何新闻就没有「为什么动」的依据，交给归因层，避免模型凭空编造
            logger.info("[行业异动] action=llm_reason status=skipped reason=no_news")
            return {}

        prompt = self._build_prompt(
            entries,
            market_news_text=market_news_text,
            sector_news=sector_news,
            catalyst_context=catalyst_context,
            language=language,
        )
        try:
            response = self.analyzer.generate_text(
                prompt,
                max_tokens=self._LLM_MAX_TOKENS,
                temperature=self._LLM_TEMPERATURE,
            )
        except Exception as exc:  # pragma: no cover - generate_text 内部已兜底
            logger.warning("[行业异动] action=llm_reason status=failed error=%s", exc)
            return {}

        if not response:
            logger.info("[行业异动] action=llm_reason status=empty_response")
            return {}

        reasons = self._parse_reasons(response, [_text(s.get("name")) for s, _ in entries])
        logger.info(
            "[行业异动] action=llm_reason status=success parsed=%d/%d",
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
        *,
        market_news_text: str,
        sector_news: Dict[str, Sequence[Any]],
        catalyst_context: Dict[str, Dict[str, Any]],
        language: str,
    ) -> str:
        blocks = "\n\n".join(
            self._format_sector_block(sector, direction, sector_news, catalyst_context, language)
            for sector, direction in entries
        )
        if language == "en":
            return (
                "You are an equity market recap analyst. For each industry below, write one line "
                "explaining WHY it led or lagged today.\n\n"
                "Output format (strict):\n"
                "- One industry per line, exactly: Industry => Reason\n"
                "- Output only those lines. No tables, numbering, headings or commentary.\n\n"
                "Content rules:\n"
                f"- Each reason must be under {self.MAX_REASON_CHARS_EN} characters and name the "
                "actual catalyst: policy or regulation, company/industry news, overnight overseas "
                "moves, fund rotation, or a leading stock.\n"
                "- Do NOT simply restate the percentage change or advancer/decliner counts. "
                "A reason that only describes how much it moved is unacceptable.\n"
                "- You may cite the sub-industry that drove the move, and name specific stocks "
                "that hit the daily limit or fell sharply, using only the material given.\n"
                "- Use ONLY the material below. Never invent events, policies, figures or companies.\n"
                "- If an industry has no supporting evidence, write: Industry => none\n\n"
                f"[Industry material]\n{blocks}\n\n"
                f"[Market-wide news]\n{market_news_text or 'none'}\n"
            )
        return (
            "你是 A 股盘后复盘分析师。请为下面每个行业写一条「异动原因」，"
            "说明它今天为什么领涨或领跌。\n\n"
            "输出格式（严格遵守）：\n"
            "- 每行一个行业，格式为：行业名 => 原因\n"
            "- 只输出这些行，不要表格、序号、标题或额外说明\n\n"
            "内容要求：\n"
            f"- 每条原因不超过 {self.MAX_REASON_CHARS} 个字，必须写出真正的催化因素："
            "政策或监管文件、公司或产业消息、隔夜外盘表现、资金轮动、龙头股带动\n"
            "- 禁止只复述涨跌幅或涨跌家数。只说明「涨了多少 / 跌了多少」的原因视为不合格\n"
            "- 可以指出是哪个子板块带动，并点名涨停或大跌的具体个股，但只能用下面给的材料\n"
            "- 只能依据下面的【行业材料】和【市场新闻】，不得编造事件、政策、数据或公司\n"
            "- 确实找不到依据的行业，写：行业名 => 无\n\n"
            "示例（仅供参考文风与信息密度，不要照抄内容）：\n"
            "传媒 => Meta Muse 爆火带动 AI 应用行情，叠加《文化产业发展\"十五五\"规划》政策利好，"
            "新华文轩、智度股份等多股涨停\n"
            "钢铁 => 行业长期\"强供给、弱需求\"格局未改，地产新开工偏弱，高炉减产慢于需求回落，"
            "供需压力大\n\n"
            f"【行业材料】\n{blocks}\n\n"
            f"【市场新闻】\n{market_news_text or '暂无'}\n"
        )

    def _format_sector_block(
        self,
        sector: Dict[str, Any],
        direction: str,
        sector_news: Dict[str, Sequence[Any]],
        catalyst_context: Dict[str, Dict[str, Any]],
        language: str,
    ) -> str:
        name = _text(sector.get("name")) or "-"
        label = ("Leading" if direction == "up" else "Lagging") if language == "en" else (
            "领涨" if direction == "up" else "领跌"
        )
        lines = [f"{label} {name} {_fmt_pct(sector.get('change_pct'))}"]

        material = catalyst_context.get(name) or {}

        subs = [s for s in (material.get("sub_sectors") or []) if isinstance(s, dict)]
        if subs:
            text = "、".join(
                f"{_text(s.get('name'))} {_fmt_pct(s.get('change_pct'))}"
                for s in subs[: self._MAX_SUB_SECTORS]
                if _text(s.get("name"))
            )
            if text:
                lines.append(f"  {'sub-industries' if language == 'en' else '子板块'}: {text}")

        limit_ups = [s for s in (material.get("limit_up_stocks") or []) if isinstance(s, dict)]
        if limit_ups:
            items = []
            for s in limit_ups[: self._MAX_STOCK_EXAMPLES]:
                stock_name = _text(s.get("name"))
                if not stock_name:
                    continue
                sub = _text(s.get("sub_industry"))
                streak = s.get("streak")
                extra = "/".join(p for p in (sub, f"{streak}板" if streak else "") if p)
                items.append(f"{stock_name}({extra})" if extra else stock_name)
            if items:
                key = "limit-up" if language == "en" else "涨停股"
                lines.append(f"  {key}: {'、'.join(items)}")

        for material_key, zh_key, en_key in (
            ("top_gainers", "领涨个股", "top gainers"),
            ("top_losers", "领跌个股", "top losers"),
        ):
            rows = [s for s in (material.get(material_key) or []) if isinstance(s, dict)]
            if not rows:
                continue
            text = "、".join(
                f"{_text(s.get('name'))} {_fmt_pct(s.get('change_pct'))}"
                for s in rows[: self._MAX_STOCK_EXAMPLES]
                if _text(s.get("name"))
            )
            if text:
                lines.append(f"  {en_key if language == 'en' else zh_key}: {text}")

        own_news = self._format_news(
            sector_news.get(name), language, limit=self._MAX_SECTOR_NEWS, indent="    "
        )
        if own_news:
            key = "related news" if language == "en" else "相关新闻"
            lines.append(f"  {key}:\n{own_news}")

        return "\n".join(lines)

    def _format_news(
        self,
        news: Optional[Sequence[Any]],
        language: str,
        *,
        limit: int,
        indent: str = "",
    ) -> str:
        lines: List[str] = []
        for item in list(news or [])[:limit]:
            title = _shorten(_news_field(item, "title"), self._NEWS_TITLE_LIMIT)
            if not title:
                continue
            source = _shorten(_news_field(item, "source"), 30)
            snippet = _shorten(_news_field(item, "snippet"), self._NEWS_SNIPPET_LIMIT)
            head = f"{indent}{len(lines) + 1}. {title}"
            if source:
                head += f" ({source})" if language == "en" else f"（{source}）"
            lines.append(f"{head}\n{indent}   {snippet}" if snippet else head)
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
        """把模型给出的行业名对回榜单里的行业名，歧义时放弃以免错配。"""
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
