"""
CorrectionEngine — LLM rewriter for soft-flagged outputs.

Spec: OPENSTINGER_GRADIENT_IMPLEMENTATION_GUIDE_V1.md §CorrectionEngine

Called when GradientInterceptor verdict is soft_flag.
Rewrites the output and re-evaluates.
If re-evaluation still fails → returns original text with soft_flag kept.
"""

from __future__ import annotations

import logging
from typing import Any

from openstinger.gradient.alignment_profile import AlignmentProfile

logger = logging.getLogger(__name__)

CORRECT_SYSTEM = """You are an agent response alignment corrector.
Given an agent's IDENTITY PROFILE, a list of ALIGNMENT ISSUES with a response,
and the ORIGINAL RESPONSE, rewrite the response to:
  - Address the alignment issues
  - Preserve the core helpful intent
  - Maintain the same factual content
  - Be no longer than the original

Return ONLY the corrected response text, no explanation."""


def _build_correct_user(
    profile: AlignmentProfile, issues: list[str], original: str
) -> str:
    issues_str = "\n".join(f"- {i}" for i in issues)
    return (
        f"IDENTITY PROFILE:\n{profile.identity_context()}\n\n"
        f"ALIGNMENT ISSUES:\n{issues_str}\n\n"
        f"ORIGINAL RESPONSE:\n{original[:3000]}\n\n"
        f"Rewrite to address the issues while preserving helpful content:"
    )


class CorrectionEngine:
    """
    LLM-based response rewriter + re-evaluation loop.
    Max 1 correction attempt (no infinite loops).
    """

    def __init__(self, llm: Any, interceptor: Any) -> None:
        self.llm = llm
        self.interceptor = interceptor  # For re-evaluation

    async def correct(
        self,
        original_text: str,
        issues: list[str],
        profile: AlignmentProfile,
        alignment_event_uuid: str | None = None,
    ) -> tuple[str, bool]:
        """
        Attempt to correct original_text.
        Returns (corrected_text, re_eval_passed).
        """
        import hashlib
        try:
            corrected = await self.llm.complete(
                system=CORRECT_SYSTEM,
                user=_build_correct_user(profile, issues, original_text),
            )
        except Exception as exc:
            logger.warning("CorrectionEngine: rewrite failed: %s", exc)
            return original_text, False

        if not corrected.strip():
            return original_text, False

        # Re-evaluate with correction engine temporarily disabled (prevents infinite recursion).
        # Spec: max 1 correction attempt — the re-evaluation never triggers another rewrite.
        original_engine = self.interceptor.correction_engine
        self.interceptor.correction_engine = None
        try:
            re_result = await self.interceptor._full_evaluate(corrected, profile)
        finally:
            self.interceptor.correction_engine = original_engine
        passed = re_result.verdict == "pass"

        logger.info(
            "CorrectionEngine: re-eval verdict=%s (original issues: %s)",
            re_result.verdict, issues[:2],
        )

        # Persist correction log
        try:
            db = getattr(self.interceptor, "db", None)
            ns = getattr(self.interceptor, "agent_namespace", "default")
            if db is not None:
                orig_hash = hashlib.sha256(original_text.encode()).hexdigest()
                corr_hash = hashlib.sha256(corrected.encode()).hexdigest()
                await db.log_correction(
                    agent_namespace=ns,
                    alignment_event_uuid=alignment_event_uuid or "",
                    original_text_hash=orig_hash,
                    corrected_text_hash=corr_hash,
                    re_eval_verdict=re_result.verdict,
                    issues=issues,
                    succeeded=passed,
                )
        except Exception as exc:
            logger.debug("CorrectionEngine: failed to log correction: %s", exc)

        return corrected if passed else original_text, passed
