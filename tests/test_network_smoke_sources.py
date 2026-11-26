# -*- coding: utf-8 -*-
"""
外部数据源与搜索链路的网络烟雾测试。

为什么单独加这个文件：network-smoke.yml 的第一步是 `pytest -m network`，但全仓
3373 个测试里只有 2 个带 `network` marker（tests/test_anspire_search.py），且都被
`@unittest.skipIf` 按 Anspire key 门控 —— 没配 key 就是 `2 skipped`，实际执行数 0。
2026-11-26 实测 run #78：该步 11 秒"通过"，而同一次运行的 quick smoke 日志里
三类情报搜索（最新消息/机构分析/风险排查）全部失败，烟雾测试一个都没报出来。

这里补上真正会打外网的断言。放在新文件而不是给既有测试补 marker，是为了尽量不和
上游 ZhuLinsen/daily_stock_analysis 产生合并冲突。

network-smoke.yml 里这一步是 continue-on-error，所以失败只会留在日志和 artifact
里、不会阻塞；要的就是"源挂了能在日志里一眼看到"。
"""

import os
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TIMEOUT = 15
UA = {"User-Agent": "Mozilla/5.0 (compatible; network-smoke)"}


@pytest.mark.network
def test_tencent_quote_reachable():
    """腾讯行情（TencentFetcher 的数据源，优先级 P0）。"""
    r = requests.get("https://qt.gtimg.cn/q=sh000001", headers=UA, timeout=TIMEOUT)
    assert r.status_code == 200, f"腾讯行情 HTTP {r.status_code}"
    assert "v_sh000001=" in r.text, f"腾讯行情返回体异常: {r.text[:120]!r}"


@pytest.mark.network
def test_eastmoney_quote_reachable():
    """东方财富 push2 行情（efinance 的底层接口）。

    run #78 日志里 `[efinance] 获取板块排行失败: Expecting value: line 1 column 1`
    就是这个域名返回了非 JSON，这条断言能把它显式暴露出来。
    """
    r = requests.get(
        "https://push2.eastmoney.com/api/qt/stock/get",
        params={"secid": "1.000001", "fields": "f43,f57,f58"},
        headers=UA,
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, f"东财 push2 HTTP {r.status_code}"
    payload = r.json()          # 返回非 JSON 时这里会抛，正是我们想抓的情况
    assert payload.get("data"), f"东财 push2 无 data 字段: {str(payload)[:160]}"


@pytest.mark.network
def test_search_service_has_available_provider():
    """至少要有一个可用的搜索 provider。

    只配 GEMINI_API_KEY、没配任何搜索 key 时，provider 只剩 SearXNG 公共实例自动
    发现模式，而公共实例对 GitHub runner 的共享出口 IP 普遍 503 / 限流。
    """
    from src.search_service import get_search_service

    service = get_search_service()
    assert service.is_available, (
        "没有可用的搜索 provider。配置任一搜索 key 即可修复（按代码里的优先级）："
        "BOCHA_API_KEYS / TAVILY_API_KEYS / BRAVE_API_KEYS / SERPAPI_API_KEYS / "
        "MINIMAX_API_KEYS / ANSPIRE_API_KEYS，或用 SEARXNG_BASE_URLS 指定自建实例。"
    )


@pytest.mark.network
def test_search_stock_news_returns_results():
    """情报搜索要能真的取到结果 —— 这条就是 run #78 漏掉的那个故障。

    用上证指数成分里最常被报道的标的做查询，正常情况下不该是 0 条。
    """
    from src.search_service import get_search_service

    service = get_search_service()
    if not service.is_available:
        pytest.skip("无可用搜索 provider，已由 test_search_service_has_available_provider 覆盖")

    resp = service.search_stock_news("600519", "贵州茅台", max_results=3)
    assert resp.success, (
        f"情报搜索失败 provider={resp.provider} error={resp.error_message}。"
        "若全是 SearXNG 公共实例的 503 / Too Many Requests，请配置一个搜索 API key。"
    )
    assert resp.results, (
        f"情报搜索返回 0 条结果 provider={resp.provider}。"
        "报告里的消息面/舆情部分会因此为空。"
    )
