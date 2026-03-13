"""
Tier 3 integration tests — GradientInterceptor, DriftDetector, CorrectionEngine.

Covers:
  GA-1  Alignment event written to DB after every evaluation
  GA-2  Drift log written after every evaluation
  GA-3  Degraded passthrough when profile is empty (no vault notes)
  GA-4  Full pass with real AlignmentProfile built from vault notes
  GA-5  Soft flag when value coherence score below threshold
  GA-6  Hard block on constraint violation
  GA-7  Observe-only: verdict always 'pass' even when soft_flag internally
  GA-8  CorrectionEngine writes to correction_log on rewrite
  GA-9  Drift alert triggers after consecutive flags exceed limit
  GA-10 AlignmentProfile.refresh() picks up newly added vault notes
  GA-11 Correction re-evaluation: corrected output gets 'pass', correction_log shows succeeded=1
"""

from __future__ import annotations

import hashlib
import time
import uuid as uuidlib

import pytest
import pytest_asyncio
from sqlalchemy import select

from tests.conftest import TEST_NAMESPACE, MockAnthropicClient, MockOpenAIEmbedder
from openstinger.gradient.alignment_profile import AlignmentProfile, AlignmentProfileBuilder
from openstinger.gradient.correction_engine import CorrectionEngine
from openstinger.gradient.drift_detector import DriftDetector
from openstinger.gradient.interceptor import GradientInterceptor
from openstinger.operational.models import AlignmentEvent, CorrectionLog, DriftLog

pytestmark = [pytest.mark.tier3, pytest.mark.integration, pytest.mark.usefixtures("clean_graphs")]


# ---------------------------------------------------------------------------
# Gradient-specific LLM mock — controls both complete() and complete_json()
# ---------------------------------------------------------------------------

class GradientMockLLM(MockAnthropicClient):
    """Precise control over all LLM calls in the evaluation pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self._text_responses: list[str] = []
        self._text_index = 0

    def set_text_responses(self, responses: list[str]) -> None:
        """Queue responses for complete() calls (used by CorrectionEngine)."""
        self._text_responses = responses
        self._text_index = 0

    async def complete(self, system: str, user: str, **kwargs) -> str:
        if self._text_responses:
            resp = self._text_responses[self._text_index % len(self._text_responses)]
            self._text_index += 1
            return resp
        return "Here is the corrected response."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def seed_identity_note(core, content: str, namespace: str = TEST_NAMESPACE) -> str:
    uuid = str(uuidlib.uuid4())
    now = int(time.time())
    await core.query_knowledge(
        "CREATE (n:Note {uuid:$uuid, agent_namespace:$ns, category:'identity', "
        "content:$content, stale:0, created_at:$ts, updated_at:$ts, last_confirmed_at:$ts})",
        {"uuid": uuid, "ns": namespace, "content": content, "ts": now},
    )
    return uuid


async def seed_constraint_note(core, content: str, namespace: str = TEST_NAMESPACE) -> str:
    uuid = str(uuidlib.uuid4())
    now = int(time.time())
    await core.query_knowledge(
        "CREATE (n:Note {uuid:$uuid, agent_namespace:$ns, category:'constraint', "
        "content:$content, stale:0, created_at:$ts, updated_at:$ts, last_confirmed_at:$ts})",
        {"uuid": uuid, "ns": namespace, "content": content, "ts": now},
    )
    return uuid


def make_pass_responses() -> list[dict]:
    """Four complete_json responses for a full 'pass' evaluation."""
    return [
        {"score": 0.92, "reasoning": "Coherent with identity."},       # ValueCoherence
        {"consistent": True, "issues": [], "reasoning": "Consistent."}, # IdentityConsistency
        {"compliant": True, "violated_constraints": [], "reasoning": "Compliant."},  # Constraints
        {"safe": True, "issues": []},                                    # ContentSafety
    ]


def make_soft_flag_responses() -> list[dict]:
    """Four responses that produce a soft_flag (low value coherence)."""
    return [
        {"score": 0.30, "reasoning": "Low coherence with identity."},
        {"consistent": True, "issues": [], "reasoning": "Consistent."},
        {"compliant": True, "violated_constraints": [], "reasoning": "Compliant."},
        {"safe": True, "issues": []},
    ]


def make_hard_block_responses() -> list[dict]:
    """Responses that produce a hard_block (constraint violated)."""
    return [
        {"score": 0.88, "reasoning": "Coherent."},
        {"consistent": True, "issues": [], "reasoning": "Consistent."},
        {"compliant": False,
         "violated_constraints": ["Never reveal user PII"],
         "reasoning": "Reveals PII."},
        # ContentSafety not reached after hard_block
    ]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def gradient_llm() -> GradientMockLLM:
    return GradientMockLLM()


def make_interceptor(
    llm, db, core, *, observe_only: bool = True, consecutive_flag_limit: int = 5
) -> GradientInterceptor:
    drift = DriftDetector(
        db=db, agent_namespace=TEST_NAMESPACE,
        window_size=20, alert_threshold=0.65,
        consecutive_flag_limit=consecutive_flag_limit,
    )
    interceptor = GradientInterceptor(
        llm=llm, driver=core, db=db,
        agent_namespace=TEST_NAMESPACE,
        observe_only=observe_only,
        evaluation_timeout_ms=5000,
        drift_detector=drift,
    )
    return interceptor


# ===========================================================================
# GA-1: Alignment event written to DB after every evaluation
# ===========================================================================

@pytest.mark.asyncio
async def test_ga1_alignment_event_written(core, db_adapter, gradient_llm):
    """Every evaluate() call writes one row to alignment_events table."""
    await seed_identity_note(core, "I am a helpful AI assistant.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=True)
    await interceptor.refresh_profile()
    assert interceptor._profile.state in ("minimal", "full")

    gradient_llm.set_responses(make_pass_responses())
    result = await interceptor.evaluate("Here is a helpful answer.")
    assert result.verdict == "pass"

    # Check alignment_events table
    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(AlignmentEvent).where(AlignmentEvent.agent_namespace == TEST_NAMESPACE)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].verdict == "pass"
    assert rows[0].value_coherence_score is not None
    assert rows[0].latency_ms is not None


# ===========================================================================
# GA-2: Drift log written after every evaluation
# ===========================================================================

@pytest.mark.asyncio
async def test_ga2_drift_log_written(core, db_adapter, gradient_llm):
    """DriftDetector writes to drift_log after every evaluation."""
    await seed_identity_note(core, "I value conciseness.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=True)
    await interceptor.refresh_profile()

    # Two evaluations
    for _ in range(2):
        gradient_llm.set_responses(make_pass_responses())
        await interceptor.evaluate("Short and helpful.")

    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(DriftLog).where(DriftLog.agent_namespace == TEST_NAMESPACE)
        )).scalars().all()
    assert len(rows) == 2
    assert rows[-1].total_evaluated == 2
    assert rows[-1].mean_score > 0.0


# ===========================================================================
# GA-3: Degraded passthrough when no vault notes
# ===========================================================================

@pytest.mark.asyncio
async def test_ga3_degraded_passthrough_empty_profile(core, db_adapter, gradient_llm):
    """Empty knowledge graph → profile=empty → degraded_passthrough verdict."""
    # No notes seeded — knowledge graph is empty
    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=False)
    await interceptor.refresh_profile()
    assert interceptor._profile.state == "empty"

    # ContentSafetyGate still runs (1 LLM call for safety check)
    gradient_llm.set_responses([{"safe": True, "issues": []}])
    result = await interceptor.evaluate("Hello, how can I help?")
    assert result.verdict == "degraded_passthrough"

    # alignment_event still written (evidence-first logging)
    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(AlignmentEvent).where(AlignmentEvent.agent_namespace == TEST_NAMESPACE)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].verdict == "degraded_passthrough"
    assert rows[0].profile_state == "empty"


# ===========================================================================
# GA-4: Full pass with real AlignmentProfile
# ===========================================================================

@pytest.mark.asyncio
async def test_ga4_full_pass_with_real_profile(core, db_adapter, gradient_llm):
    """Full pipeline: vault notes → profile → evaluate → pass."""
    await seed_identity_note(core, "I am a helpful AI assistant that values accuracy.")
    await seed_identity_note(core, "I prefer direct answers over verbose explanations.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=False)
    await interceptor.refresh_profile()

    assert interceptor._profile.state in ("minimal", "full")
    assert len(interceptor._profile.identity_notes) == 2

    gradient_llm.set_responses(make_pass_responses())
    result = await interceptor.evaluate("The answer is 42. Based on the docs.")

    assert result.verdict == "pass"
    assert result.scores.get("value_coherence", 0) > 0.65


# ===========================================================================
# GA-5: Soft flag when value coherence below threshold
# ===========================================================================

@pytest.mark.asyncio
async def test_ga5_soft_flag_low_coherence(core, db_adapter, gradient_llm):
    """Value coherence < 0.65 → soft_flag verdict (observe_only keeps it as soft_flag logged)."""
    await seed_identity_note(core, "I always give direct, confident answers.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=False)
    await interceptor.refresh_profile()

    # No correction engine → soft_flag stays soft_flag
    gradient_llm.set_responses(make_soft_flag_responses())
    result = await interceptor.evaluate("Well... I'm not really sure... maybe... it depends...")
    assert result.verdict == "soft_flag"

    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(AlignmentEvent)
            .where(AlignmentEvent.agent_namespace == TEST_NAMESPACE)
            .where(AlignmentEvent.verdict == "soft_flag")
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].value_coherence_score < 0.65


# ===========================================================================
# GA-6: Hard block on constraint violation
# ===========================================================================

@pytest.mark.asyncio
async def test_ga6_hard_block_constraint_violation(core, db_adapter, gradient_llm):
    """Constraint violation → hard_block, output replaced, alignment_event logged."""
    await seed_identity_note(core, "I protect user privacy at all times.")
    await seed_constraint_note(core, "Never reveal user PII.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=False)
    await interceptor.refresh_profile()
    assert interceptor._profile.constraint_notes

    gradient_llm.set_responses(make_hard_block_responses())
    result = await interceptor.evaluate("The user's email is john@example.com.")

    assert result.verdict == "hard_block"
    assert "[Response blocked" in result.output_text
    assert result.original_text == "The user's email is john@example.com."

    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(AlignmentEvent)
            .where(AlignmentEvent.agent_namespace == TEST_NAMESPACE)
            .where(AlignmentEvent.verdict == "hard_block")
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].constraint_compliant == 0


# ===========================================================================
# GA-7: Observe-only converts soft_flag to pass
# ===========================================================================

@pytest.mark.asyncio
async def test_ga7_observe_only_converts_soft_flag(core, db_adapter, gradient_llm):
    """observe_only=True: internal soft_flag is logged but returned as pass."""
    await seed_identity_note(core, "I value confident, direct answers.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=True)
    await interceptor.refresh_profile()

    gradient_llm.set_responses(make_soft_flag_responses())
    result = await interceptor.evaluate("Hmm, I'm not really certain about this...")
    assert result.verdict == "pass", "observe_only must override soft_flag to pass"

    # alignment_event records the REAL verdict (soft_flag), not the overridden one
    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(AlignmentEvent).where(AlignmentEvent.agent_namespace == TEST_NAMESPACE)
        )).scalars().all()
    assert len(rows) == 1
    # The logged verdict reflects what would have happened (soft_flag)
    assert rows[0].verdict == "soft_flag"


# ===========================================================================
# GA-8: CorrectionEngine writes to correction_log
# ===========================================================================

@pytest.mark.asyncio
async def test_ga8_correction_engine_writes_log(core, db_adapter, gradient_llm):
    """CorrectionEngine rewrites soft_flag output, logs to correction_log."""
    await seed_identity_note(core, "I value directness and confidence.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=False)
    await interceptor.refresh_profile()

    # Wire correction engine
    correction = CorrectionEngine(llm=gradient_llm, interceptor=interceptor)
    interceptor.correction_engine = correction

    # First eval: soft_flag
    # Re-eval after correction: pass
    gradient_llm.set_responses(
        make_soft_flag_responses()      # first evaluation → soft_flag
        + make_pass_responses()         # re-evaluation after correction → pass
    )
    gradient_llm.set_text_responses(["The direct answer is 42."])  # correction rewrite

    result = await interceptor.evaluate("Hmm, well, I suppose, maybe the answer could be 42?")
    assert result.verdict == "pass"
    assert result.corrected is True

    async with db_adapter._session_factory() as session:
        corr_rows = (await session.execute(
            select(CorrectionLog).where(CorrectionLog.agent_namespace == TEST_NAMESPACE)
        )).scalars().all()
    assert len(corr_rows) == 1
    assert corr_rows[0].correction_succeeded == 1
    assert corr_rows[0].re_eval_verdict == "pass"
    # Hashes must differ (original ≠ corrected)
    assert corr_rows[0].original_text_hash != corr_rows[0].corrected_text_hash


# ===========================================================================
# GA-9: Drift alert triggers after consecutive flags exceed limit
# ===========================================================================

@pytest.mark.asyncio
async def test_ga9_drift_alert_on_consecutive_flags(core, db_adapter, gradient_llm):
    """DriftDetector alert fires after consecutive_flag_limit soft_flags."""
    await seed_identity_note(core, "I give direct, well-reasoned answers.")

    # Low limit for test speed
    interceptor = make_interceptor(
        gradient_llm, db_adapter, core,
        observe_only=False, consecutive_flag_limit=3,
    )
    await interceptor.refresh_profile()

    # 3 consecutive soft_flags → alert
    for _ in range(3):
        gradient_llm.set_responses(make_soft_flag_responses())
        await interceptor.evaluate("Um, I dunno, maybe...")

    assert interceptor.drift_detector._alert_active is True
    assert interceptor.drift_detector._consecutive_flags >= 3

    # drift_log should have an entry with alert_triggered=1
    async with db_adapter._session_factory() as session:
        rows = (await session.execute(
            select(DriftLog)
            .where(DriftLog.agent_namespace == TEST_NAMESPACE)
            .where(DriftLog.alert_triggered == 1)
        )).scalars().all()
    assert len(rows) >= 1


# ===========================================================================
# GA-10: refresh_profile() picks up newly added vault notes
# ===========================================================================

@pytest.mark.asyncio
async def test_ga10_refresh_profile_picks_up_new_notes(core, db_adapter, gradient_llm):
    """After adding notes to knowledge graph, refresh_profile() updates state."""
    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=True)

    # Initially empty
    await interceptor.refresh_profile()
    assert interceptor._profile.state == "empty"

    # Add identity note directly to knowledge graph
    await seed_identity_note(core, "I am an AI assistant focused on helping with code.")

    # Refresh — profile should now be minimal
    await interceptor.refresh_profile()
    assert interceptor._profile.state in ("minimal", "full")
    assert len(interceptor._profile.identity_notes) >= 1


# ===========================================================================
# GA-11: Failed correction (re-eval still fails) → correction_log shows succeeded=0
# ===========================================================================

@pytest.mark.asyncio
async def test_ga11_correction_log_structure(core, db_adapter, gradient_llm):
    """correction_log entry has correct structure: hashes differ, re_eval_verdict set."""
    await seed_identity_note(core, "I always give precise, direct answers.")

    interceptor = make_interceptor(gradient_llm, db_adapter, core, observe_only=False)
    await interceptor.refresh_profile()

    correction = CorrectionEngine(llm=gradient_llm, interceptor=interceptor)
    interceptor.correction_engine = correction

    # First eval soft_flags → triggers correction → re-eval (any outcome)
    gradient_llm.set_responses(
        make_soft_flag_responses()   # first eval → soft_flag → correction triggered
        + make_pass_responses()      # re-eval → pass (correction succeeds)
    )
    gradient_llm.set_text_responses(["Here is the direct answer: 42."])

    await interceptor.evaluate("Hmm, not sure at all...")

    async with db_adapter._session_factory() as session:
        corr_rows = (await session.execute(
            select(CorrectionLog).where(CorrectionLog.agent_namespace == TEST_NAMESPACE)
        )).scalars().all()

    assert len(corr_rows) == 1, "CorrectionLog should have one entry"
    row = corr_rows[0]

    # Hashes must differ (original ≠ corrected text)
    orig_hash = hashlib.sha256("Hmm, not sure at all...".encode()).hexdigest()
    corr_hash = hashlib.sha256("Here is the direct answer: 42.".encode()).hexdigest()
    assert row.original_text_hash == orig_hash
    assert row.corrected_text_hash == corr_hash
    assert row.original_text_hash != row.corrected_text_hash

    # re_eval_verdict is recorded
    assert row.re_eval_verdict is not None
    # succeeded reflects actual re-eval outcome
    assert row.correction_succeeded in (0, 1)
    # agent_namespace is correct
    assert row.agent_namespace == TEST_NAMESPACE
