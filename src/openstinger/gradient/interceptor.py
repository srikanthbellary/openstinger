"""
GradientInterceptor — synchronous evaluation pipeline (Tier 3 core).

Spec: OPENSTINGER_GRADIENT_IMPLEMENTATION_GUIDE_V1.md §GradientInterceptor

Evaluates every agent output BEFORE delivery via 4 dimensions:
  1. ValueCoherenceScorer      (haiku, fast)
  2. IdentityConsistencyCheck  (haiku, fast)
  3. ConstraintComplianceCheck (haiku, fast)
  4. ContentSafetyGate         (pattern + haiku, always runs)

Verdicts:
  pass                 — all checks pass
  soft_flag            — below threshold, CorrectionEngine invoked
  hard_block           — constraint violated, output blocked
  timeout_passthrough  — evaluation timed out → pass through
  degraded_passthrough — FalkorDB unreachable, vault empty → safety-only

Calibration:
  observe_only=True  → evaluate and log but NEVER block/correct
  observe_only=False → active correction and blocking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from openstinger.gradient.alignment_profile import AlignmentProfile, AlignmentProfileBuilder
from openstinger.gradient.evaluators.value_coherence import ValueCoherenceScorer
from openstinger.gradient.evaluators.identity_consistency import IdentityConsistencyCheck
from openstinger.gradient.evaluators.constraint_compliance import ConstraintComplianceCheck
from openstinger.gradient.evaluators.content_safety import ContentSafetyGate

logger = logging.getLogger(__name__)

Verdict = Literal[
    "pass", "soft_flag", "hard_block",
    "timeout_passthrough", "degraded_passthrough"
]

VALUE_COHERENCE_THRESHOLD = 0.65   # below → soft_flag


@dataclass
class EvaluationResult:
    verdict: Verdict
    output_text: str               # possibly corrected output
    original_text: str
    scores: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    evaluated_at: int = field(default_factory=lambda: int(time.time()))
    latency_ms: int = 0
    corrected: bool = False


class GradientInterceptor:
    """
    Synchronous evaluation pipeline. Call evaluate() before delivering
    any agent output.
    """

    def __init__(
        self,
        llm: Any,
        driver: Any,
        db: Any,
        agent_namespace: str = "default",
        observe_only: bool = True,
        evaluation_timeout_ms: int = 2000,
        value_threshold: float = VALUE_COHERENCE_THRESHOLD,
        correction_engine: Any = None,
        drift_detector: Any = None,
    ) -> None:
        self.llm = llm
        self.driver = driver
        self.db = db
        self.agent_namespace = agent_namespace
        self.observe_only = observe_only
        self.evaluation_timeout_ms = evaluation_timeout_ms
        self.value_threshold = value_threshold

        # Evaluators
        self.value_scorer = ValueCoherenceScorer(llm)
        self.identity_check = IdentityConsistencyCheck(llm)
        self.constraint_check = ConstraintComplianceCheck(llm)
        self.safety_gate = ContentSafetyGate(llm)

        # Optional components
        self.correction_engine = correction_engine
        self.drift_detector = drift_detector

        # Profile (refreshed on vault sync)
        self._profile: Optional[AlignmentProfile] = None
        self._profile_builder: Optional[AlignmentProfileBuilder] = None

        if driver:
            self._profile_builder = AlignmentProfileBuilder(driver, agent_namespace)

    async def refresh_profile(self) -> None:
        """Rebuild AlignmentProfile from vault. Call after each vault sync."""
        if self._profile_builder:
            try:
                self._profile = await self._profile_builder.build()
                logger.info(
                    "AlignmentProfile refreshed: state=%s", self._profile.state
                )
            except Exception as exc:
                logger.warning("Profile refresh failed: %s", exc)
                self._profile = None

    async def evaluate(self, output_text: str) -> EvaluationResult:
        """
        Evaluate output_text synchronously before delivery.

        Returns EvaluationResult. If observe_only=True, verdict is always
        'pass' (but the real verdict is logged).
        """
        start_ms = int(time.time() * 1000)

        # Degraded mode: profile not available OR not yet usable (empty/bootstrapping)
        # Spec (09_ALIGNMENT_PROFILE_BOOTSTRAPPING_GUIDE.md):
        #   "empty" and "bootstrapping" states → safety-only mode → degraded_passthrough
        if self._profile is None or not self._profile.is_usable:
            result = await self._degraded_evaluate(output_text)
            result.latency_ms = int(time.time() * 1000) - start_ms
            await self._log_event(result)
            return result

        # Full evaluation with timeout
        try:
            result = await asyncio.wait_for(
                self._full_evaluate(output_text, self._profile),
                timeout=self.evaluation_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "GradientInterceptor: evaluation timed out (%dms)", self.evaluation_timeout_ms
            )
            result = EvaluationResult(
                verdict="timeout_passthrough",
                output_text=output_text,
                original_text=output_text,
            )

        result.latency_ms = int(time.time() * 1000) - start_ms

        # Log first — get event UUID for correction linking
        event_uuid = await self._log_event(result)
        result.event_uuid = event_uuid  # type: ignore[attr-defined]

        # Drift detection
        if self.drift_detector:
            await self.drift_detector.record(result)

        # If observe_only, never actually block/correct
        if self.observe_only and result.verdict in ("soft_flag", "hard_block"):
            logger.info(
                "GradientInterceptor [observe_only]: would have %s — passing through",
                result.verdict,
            )
            result.verdict = "pass"

        return result

    async def _full_evaluate(
        self, output_text: str, profile: AlignmentProfile
    ) -> EvaluationResult:
        """Run all 4 evaluation dimensions."""
        issues: list[str] = []
        scores: dict = {}

        # --- Dimension 1: Value coherence ---
        vc = await self.value_scorer.score(output_text, profile)
        scores["value_coherence"] = vc.get("score", 1.0)
        if not vc.get("skipped") and vc["score"] < self.value_threshold:
            issues.append(f"value_coherence_low: {vc.get('reasoning', '')}")

        # --- Dimension 2: Identity consistency ---
        ic = await self.identity_check.check(output_text, profile)
        scores["identity_consistent"] = ic.get("consistent", True)
        if not ic.get("skipped") and not ic["consistent"]:
            issues.extend(ic.get("issues", []))

        # --- Dimension 3: Constraint compliance ---
        cc = await self.constraint_check.check(output_text, profile)
        scores["constraint_compliant"] = cc.get("compliant", True)
        if not cc.get("skipped") and not cc["compliant"]:
            # Hard block — constraint violated
            violated = cc.get("violated_constraints", [])
            return EvaluationResult(
                verdict="hard_block",
                output_text="[Response blocked: constraint violation]",
                original_text=output_text,
                scores=scores,
                issues=violated,
            )

        # --- Dimension 4: Content safety (always) ---
        sg = await self.safety_gate.check(output_text)
        scores["content_safe"] = sg.get("safe", True)
        if not sg["safe"]:
            return EvaluationResult(
                verdict="hard_block",
                output_text="[Response blocked: content safety]",
                original_text=output_text,
                scores=scores,
                issues=sg.get("issues", []),
            )

        # Determine verdict
        if issues:
            # Soft flag — attempt correction if engine available
            if self.correction_engine and not self.observe_only:
                corrected_text, re_eval_passed = await self.correction_engine.correct(
                    output_text, issues, profile
                )
                return EvaluationResult(
                    verdict="pass" if re_eval_passed else "soft_flag",
                    output_text=corrected_text,
                    original_text=output_text,
                    scores=scores,
                    issues=issues,
                    corrected=True,
                )
            return EvaluationResult(
                verdict="soft_flag",
                output_text=output_text,
                original_text=output_text,
                scores=scores,
                issues=issues,
            )

        return EvaluationResult(
            verdict="pass",
            output_text=output_text,
            original_text=output_text,
            scores=scores,
        )

    async def _degraded_evaluate(self, output_text: str) -> EvaluationResult:
        """Safety-only evaluation when profile is unavailable."""
        sg = await self.safety_gate.check(output_text)
        if not sg["safe"]:
            return EvaluationResult(
                verdict="hard_block",
                output_text="[Response blocked: content safety (degraded mode)]",
                original_text=output_text,
                issues=sg.get("issues", []),
            )
        return EvaluationResult(
            verdict="degraded_passthrough",
            output_text=output_text,
            original_text=output_text,
        )

    async def _log_event(self, result: EvaluationResult) -> Optional[str]:
        """Persist evaluation event to alignment_events table. Returns event UUID."""
        try:
            profile_state = self._profile.state if self._profile else None
            event_uuid = await self.db.log_alignment_event(
                agent_namespace=self.agent_namespace,
                verdict=result.verdict,
                scores=result.scores,
                issues=result.issues,
                corrected=result.corrected,
                profile_state=profile_state,
                latency_ms=result.latency_ms,
            )
            logger.debug(
                "GradientInterceptor: verdict=%s latency=%dms event=%s",
                result.verdict, result.latency_ms, event_uuid[:8],
            )
            return event_uuid
        except Exception as exc:
            logger.warning("Failed to log alignment event: %s", exc)
            return None
