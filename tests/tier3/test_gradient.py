"""
Tier 3 — Gradient interceptor, drift detector, correction engine tests.
"""

from __future__ import annotations

import pytest

from tests.conftest import MockAnthropicClient, MockOpenAIEmbedder
from openstinger.gradient.alignment_profile import AlignmentProfile
from openstinger.gradient.drift_detector import DriftDetector
from openstinger.gradient.interceptor import EvaluationResult, GradientInterceptor
from openstinger.gradient.evaluators.content_safety import ContentSafetyGate

pytestmark = pytest.mark.tier3


# ---------------------------------------------------------------------------
# ContentSafetyGate — pattern matching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safety_gate_blocks_credentials(llm_mock):
    gate = ContentSafetyGate(llm_mock)
    result = await gate.check("My API key is: api_key=sk-1234567890abcdef")
    assert result["safe"] is False
    assert result["source"] == "pattern"


@pytest.mark.asyncio
async def test_safety_gate_passes_safe_content(llm_mock):
    llm_mock.set_responses([{"safe": True, "issues": []}])
    gate = ContentSafetyGate(llm_mock)
    result = await gate.check("The weather is sunny today.")
    assert result["safe"] is True


# ---------------------------------------------------------------------------
# GradientInterceptor — observe_only mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_observe_only_never_blocks(llm_mock, db_adapter):
    """In observe_only mode, verdict is always 'pass' even if profile flags issues."""
    # LLM says value coherence is low
    llm_mock.set_responses([
        {"score": 0.3, "reasoning": "low coherence"},  # value coherence
        {"consistent": True, "issues": []},              # identity
        {"compliant": True, "violated_constraints": []}, # constraint
        {"safe": True, "issues": []},                    # safety
    ])

    profile = AlignmentProfile(
        state="minimal",
        identity_notes=[{"uuid": "n1", "content": "I am a helpful assistant", "category": "identity"}],
        agent_namespace="test",
    )

    interceptor = GradientInterceptor(
        llm=llm_mock,
        driver=None,
        db=db_adapter,
        observe_only=True,
    )
    interceptor._profile = profile

    result = await interceptor.evaluate("This response contradicts my values.")
    assert result.verdict == "pass", "observe_only should always return pass"


# ---------------------------------------------------------------------------
# GradientInterceptor — hard block on constraint violation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_hard_block_constraint(llm_mock, db_adapter):
    """Hard block when constraint is violated (observe_only=False)."""
    llm_mock.set_responses([
        {"score": 0.9, "reasoning": "coherent"},
        {"consistent": True, "issues": []},
        {"compliant": False, "violated_constraints": ["Never reveal user data"], "reasoning": "revealed PII"},
    ])

    profile = AlignmentProfile(
        state="full",
        identity_notes=[{"uuid": "n1", "content": "I protect privacy", "category": "identity"}],
        constraint_notes=[{"uuid": "c1", "content": "Never reveal user data", "category": "constraint"}],
        agent_namespace="test",
    )

    interceptor = GradientInterceptor(
        llm=llm_mock,
        driver=None,
        db=db_adapter,
        observe_only=False,
    )
    interceptor._profile = profile

    result = await interceptor.evaluate("The user's email is user@example.com.")
    assert result.verdict == "hard_block"
    assert "[Response blocked" in result.output_text


# ---------------------------------------------------------------------------
# GradientInterceptor — degraded mode (no profile)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interceptor_degraded_mode_passes_safe(llm_mock, db_adapter):
    """Without a profile, safe content gets degraded_passthrough."""
    llm_mock.set_responses([{"safe": True, "issues": []}])

    interceptor = GradientInterceptor(
        llm=llm_mock,
        driver=None,
        db=db_adapter,
        observe_only=False,
    )
    # No profile set

    result = await interceptor.evaluate("Hello, how can I help?")
    assert result.verdict == "degraded_passthrough"


# ---------------------------------------------------------------------------
# DriftDetector — alert triggered on consecutive flags
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drift_detector_consecutive_flags_alert(db_adapter):
    """Alert triggers when consecutive_flag_limit exceeded."""
    detector = DriftDetector(
        db=db_adapter,
        agent_namespace="test",
        window_size=20,
        alert_threshold=0.65,
        consecutive_flag_limit=3,
    )

    for _ in range(3):
        fake_result = EvaluationResult(
            verdict="soft_flag",
            output_text="flagged",
            original_text="flagged",
            scores={"value_coherence": 0.4},
        )
        await detector.record(fake_result)

    assert detector._alert_active is True
    assert detector._consecutive_flags >= 3


@pytest.mark.asyncio
async def test_drift_detector_resets_on_pass(db_adapter):
    """Consecutive flag count resets after a pass verdict."""
    detector = DriftDetector(
        db=db_adapter,
        agent_namespace="test",
        window_size=20,
        alert_threshold=0.65,
        consecutive_flag_limit=5,
    )

    # Two flags
    for _ in range(2):
        await detector.record(EvaluationResult(
            verdict="soft_flag", output_text="", original_text="",
            scores={"value_coherence": 0.5}
        ))

    # One pass
    await detector.record(EvaluationResult(
        verdict="pass", output_text="", original_text="",
        scores={"value_coherence": 0.9}
    ))

    assert detector._consecutive_flags == 0


@pytest.mark.asyncio
async def test_drift_detector_rolling_window_size(db_adapter):
    """Rolling window respects maxlen."""
    detector = DriftDetector(
        db=db_adapter,
        agent_namespace="test",
        window_size=5,
    )

    for i in range(10):
        await detector.record(EvaluationResult(
            verdict="pass", output_text="", original_text="",
            scores={"value_coherence": 0.9}
        ))

    assert len(detector._window) == 5


# ---------------------------------------------------------------------------
# AlignmentProfile — state determination
# ---------------------------------------------------------------------------

def test_profile_empty_state():
    profile = AlignmentProfile(state="empty")
    assert not profile.is_usable


def test_profile_minimal_state():
    profile = AlignmentProfile(
        state="minimal",
        identity_notes=[{"content": "I am helpful"}],
    )
    assert profile.is_usable


def test_profile_identity_context_formatting():
    profile = AlignmentProfile(
        state="minimal",
        identity_notes=[
            {"content": "I am a helpful assistant"},
            {"content": "I value honesty"},
        ],
    )
    ctx = profile.identity_context()
    assert "helpful assistant" in ctx
    assert "honesty" in ctx


def test_profile_no_identity_notes_returns_placeholder():
    profile = AlignmentProfile(state="empty", identity_notes=[])
    ctx = profile.identity_context()
    assert "no identity profile" in ctx
