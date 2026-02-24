"""
DriftDetector — rolling window alignment tracker.

Spec: OPENSTINGER_GRADIENT_IMPLEMENTATION_GUIDE_V1.md §DriftDetector

Defaults:
  window_size: 20 samples
  alert_threshold: 0.65 (mean alignment score)
  consecutive_flag_limit: 5
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DriftStatus:
    """Current drift detector state."""
    window_size: int = 20
    current_window: list[float] = field(default_factory=list)
    mean_score: float = 1.0
    consecutive_flags: int = 0
    alert_active: bool = False
    total_evaluated: int = 0
    total_flagged: int = 0
    last_updated_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def soft_flag_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.total_flagged / self.total_evaluated


class DriftDetector:
    """
    Tracks rolling window of alignment scores.
    Triggers alert when mean drops below threshold or
    consecutive flags exceed limit.
    """

    def __init__(
        self,
        db: Any,
        agent_namespace: str = "default",
        window_size: int = 20,
        alert_threshold: float = 0.65,
        consecutive_flag_limit: int = 5,
    ) -> None:
        self.db = db
        self.agent_namespace = agent_namespace
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.consecutive_flag_limit = consecutive_flag_limit

        self._window: collections.deque[float] = collections.deque(maxlen=window_size)
        self._consecutive_flags = 0
        self._total_evaluated = 0
        self._total_flagged = 0
        self._alert_active = False

    async def record(self, evaluation_result: Any) -> None:
        """
        Record an evaluation result into the rolling window.
        Triggers alert if thresholds exceeded.
        """
        verdict = evaluation_result.verdict
        scores = evaluation_result.scores

        # Convert verdict to a score for the window
        score = self._verdict_to_score(verdict, scores)
        self._window.append(score)
        self._total_evaluated += 1

        if verdict in ("soft_flag", "hard_block"):
            self._consecutive_flags += 1
            self._total_flagged += 1
        else:
            self._consecutive_flags = 0

        # Check thresholds
        window_mean = sum(self._window) / len(self._window) if self._window else 1.0
        alert_triggered = (
            window_mean < self.alert_threshold
            or self._consecutive_flags >= self.consecutive_flag_limit
        )

        if alert_triggered and not self._alert_active:
            self._alert_active = True
            await self._trigger_alert(window_mean, self._consecutive_flags)
        elif not alert_triggered and self._alert_active:
            self._alert_active = False
            logger.info("DriftDetector: alert cleared (mean=%.2f)", window_mean)

        await self._persist()

    def get_status(self) -> DriftStatus:
        window_mean = sum(self._window) / len(self._window) if self._window else 1.0
        return DriftStatus(
            window_size=self.window_size,
            current_window=list(self._window),
            mean_score=round(window_mean, 3),
            consecutive_flags=self._consecutive_flags,
            alert_active=self._alert_active,
            total_evaluated=self._total_evaluated,
            total_flagged=self._total_flagged,
        )

    @staticmethod
    def _verdict_to_score(verdict: str, scores: dict) -> float:
        if verdict == "pass":
            return scores.get("value_coherence", 1.0)
        elif verdict == "soft_flag":
            return scores.get("value_coherence", 0.5) * 0.7
        elif verdict in ("hard_block",):
            return 0.0
        else:
            return 1.0  # passthrough variants don't penalise

    async def _trigger_alert(self, mean: float, consecutive: int) -> None:
        logger.warning(
            "DriftDetector ALERT: namespace=%s mean=%.2f consecutive_flags=%d",
            self.agent_namespace, mean, consecutive,
        )
        await self._persist(alert_triggered=True)

    async def _persist(self, alert_triggered: bool = False) -> None:
        """Persist rolling window state to drift_log table."""
        try:
            window_mean = sum(self._window) / len(self._window) if self._window else 1.0
            await self.db.log_drift_state(
                agent_namespace=self.agent_namespace,
                window_size=self.window_size,
                mean_score=window_mean,
                consecutive_flags=self._consecutive_flags,
                total_evaluated=self._total_evaluated,
                total_flagged=self._total_flagged,
                alert_triggered=alert_triggered or self._alert_active,
                window=list(self._window),
            )
        except Exception as exc:
            logger.debug("DriftDetector: failed to persist to drift_log: %s", exc)
