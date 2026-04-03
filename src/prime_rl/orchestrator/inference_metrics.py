from __future__ import annotations

import asyncio
import time
from collections import deque

import wandb
from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger

HistoryKey = tuple[str, str]


def parse_prometheus_text(
    text: str, gauge_names: set[str], counter_names: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Parse Prometheus exposition format text and extract metric values.

    Returns (gauges, counters) dicts keyed by metric_name.
    Aggregates across all engines by summing values.
    """
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        if family.type == "gauge" and family.name in gauge_names:
            for sample in family.samples:
                gauges[family.name] = gauges.get(family.name, 0.0) + sample.value
        elif family.type == "counter" and family.name in counter_names:
            for sample in family.samples:
                counters[family.name] = counters.get(family.name, 0.0) + sample.value
    return gauges, counters


class InferenceMetricsCollector:
    """Polls Prometheus /metrics from inference servers and logs to W&B on a time-based axis.

    Reads the current admin_clients from the inference pool on each poll, so it
    automatically adapts when servers join or leave (elastic pools).

    Metrics are collected and logged every 5 seconds, independently of training steps.
    The x-axis is wall-clock time (timestamp).

    For counter metrics (prompt/generation tokens), rates are computed between
    consecutive polls and averaged, matching vLLM's own throughput calculation.
    """

    GAUGE_METRICS = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    }

    COUNTER_METRICS = {
        "vllm:prompt_tokens",
        "vllm:generation_tokens",
    }

    COUNTER_RATE_NAMES = {
        "vllm:prompt_tokens": "prompt_throughput_tps",
        "vllm:generation_tokens": "generation_throughput_tps",
    }

    POLL_INTERVAL = 5.0

    def __init__(self, inference_pool: InferencePool, window_size: int = 20):
        self.inference_pool = inference_pool
        self.window_size = window_size
        self.logger = get_logger()
        self._history: dict[HistoryKey, deque[float]] = {}
        self._prev_counters: dict[HistoryKey, tuple[float, float]] = {}
        self._active_urls: set[str] = set()
        self._task: asyncio.Task | None = None

    async def start(self):
        wandb.define_metric("inference/*", step_metric="_timestamp")

        async def poll_loop():
            while True:
                await self._collect_and_log()
                await asyncio.sleep(self.POLL_INTERVAL)

        self._task = asyncio.create_task(poll_loop())

    async def _collect_and_log(self):
        """Fetch /metrics from all servers, update history, and log to W&B."""
        admin_clients = self.inference_pool.admin_clients
        now = time.monotonic()

        async def fetch(
            client: AsyncClient,
        ) -> tuple[str, dict[str, float] | None, dict[str, float] | None]:
            url = str(client.base_url)
            try:
                response = await client.get("/metrics", timeout=5.0)
                response.raise_for_status()
                gauges, counters = parse_prometheus_text(response.text, self.GAUGE_METRICS, self.COUNTER_METRICS)
                return url, gauges, counters
            except Exception as e:
                self.logger.debug(f"Failed to fetch metrics from {url}: {e!r}")
                return url, None, None

        results = await asyncio.gather(*[fetch(client) for client in admin_clients])

        active_urls = {str(client.base_url) for client in admin_clients}
        for key in list(self._history):
            if key[0] not in active_urls:
                del self._history[key]
        for key in list(self._prev_counters):
            if key[0] not in active_urls:
                del self._prev_counters[key]
        self._active_urls = active_urls

        for url, gauges, counters in results:
            if gauges is None:
                continue

            for metric_name, value in gauges.items():
                key = (url, metric_name)
                if key not in self._history:
                    self._history[key] = deque(maxlen=self.window_size)
                self._history[key].append(value)

            if counters is None:
                continue

            for metric_name, value in counters.items():
                key = (url, metric_name)
                prev = self._prev_counters.get(key)
                self._prev_counters[key] = (now, value)
                if prev is None:
                    continue
                prev_time, prev_value = prev
                dt = now - prev_time
                if dt <= 0:
                    continue
                rate = (value - prev_value) / dt
                rate_key = (url, self.COUNTER_RATE_NAMES[metric_name])
                if rate_key not in self._history:
                    self._history[rate_key] = deque(maxlen=self.window_size)
                self._history[rate_key].append(rate)

        metrics = self._get_metrics()
        if metrics:
            wandb.log(metrics)

    def _get_metrics(self) -> dict[str, float]:
        """Return per-server running averages and aggregates across all servers."""
        metrics: dict[str, float] = {}
        sorted_urls = sorted(self._active_urls)
        url_to_idx = {url: i for i, url in enumerate(sorted_urls)}

        aggregates: dict[str, list[float]] = {}

        for (url, metric_name), values in self._history.items():
            if url not in url_to_idx or not values:
                continue
            server_idx = url_to_idx[url]
            short_name = metric_name.removeprefix("vllm:")
            avg = sum(values) / len(values)
            metrics[f"inference/{short_name}/server_{server_idx}"] = avg
            aggregates.setdefault(short_name, []).append(avg)

        for short_name, values in aggregates.items():
            if short_name == "kv_cache_usage_perc":
                metrics[f"inference/{short_name}/mean"] = sum(values) / len(values)
            else:
                metrics[f"inference/{short_name}/mean"] = sum(values) / len(values)
                metrics[f"inference/{short_name}/sum"] = sum(values)

        return metrics

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
