"""
Agent Profile Ingester — framework-agnostic LLM heuristic configuration extractor.

Runs alongside SessionReader to watch for agent definition files (e.g., SKILL.md, 
agent.yaml, memory.json) and extracts Identity, Preferences, and Constraints 
into the FalkorDB Vault using LLM reasoning.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)

# Extensions to scan for potential identity configs
PROFILE_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}
MAX_FILE_SIZE_BYTES = 50 * 1024  # 50 KB limit


class AgentProfileIngester:
    def __init__(
        self,
        profile_dirs: list[Path],
        agent_namespace: str,
        engine: Any,
        db_adapter: Any,
        poll_interval: float = 60.0,
    ) -> None:
        self.profile_dirs = profile_dirs
        self.agent_namespace = agent_namespace
        self.engine = engine
        self.db = db_adapter
        self.poll_interval = poll_interval

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "AgentProfileIngester started for namespace %r (dirs: %s)",
            self.agent_namespace,
            [str(d) for d in self.profile_dirs],
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        await self._task
        self._task = None

    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.scan_and_ingest()
            except Exception as e:
                logger.error("AgentProfileIngester error: %s", e)
            
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    async def scan_and_ingest(self) -> int:
        """Scan configured dirs for new/modified config files and run LLM extraction."""
        files_processed = 0
        
        # Look for config files non-recursively in the dirs
        candidates = []
        for p_dir in self.profile_dirs:
            if not p_dir.exists():
                continue
            for ext in PROFILE_EXTENSIONS:
                candidates.extend(p_dir.glob(f"*{ext}"))
                
        for file_path in candidates:
            if not file_path.is_file() or file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
                
            # Ignore session files or obvious logs
            if "session" in file_path.name.lower() or "log" in file_path.name.lower():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                
                # Check if we already processed this exact file version
                # Note: Requires Tier 2 VaultChecksum or a new table.
                # Re-using VaultChecksum since it serves the exact same purpose.
                existing_hash = await self.db.get_vault_checksum(
                    self.agent_namespace, str(file_path)
                )
                
                if existing_hash == file_hash:
                    continue  # Unchanged
                    
                logger.info("Found new/modified profile file: %s", file_path.name)
                
                # Run LLM Extraction
                notes = await self.extract_components_via_llm(content, file_path.name)
                
                if notes:
                    await self._seed_vault(notes)
                    
                # Save hash so we don't process it again unless it changes
                await self.db.set_vault_checksum(
                    self.agent_namespace, str(file_path), file_hash
                )
                files_processed += 1
                
            except Exception as e:
                logger.error("Failed to process profile %s: %s", file_path.name, e)
                
        return files_processed

    async def extract_components_via_llm(self, content: str, filename: str) -> list[dict]:
        """
        Passes the raw config file to the LLM to pull out IDENTITY and CONSTRAINT notes.
        Framework agnostic — reads the intent, not the syntax.
        """
        prompt = f"""
        Analyze the following agent configuration file named '{filename}'.
        Extract any core IDENTITY definitions ("You are X", persona details), 
        behavioral PREFERENCES (how it likes to act), and hard CONSTRAINTS (rules, limitations).
        
        Ignore technical boilerplate, API keys, or JSON formatting structure.
        Focus ONLY on the agent's semantic identity and rules.
        
        Return your findings as a strict JSON array of objects. Each object must have:
        - "category": Must be one of ["identity", "preference", "constraint"]
        - "content": The actual fact or rule (e.g. "The agent is named Claudia and acts as an expert python coder.")
        
        If no relevant identity/rules are found, return an empty array [].
        
        File Content:
        ---
        {content[:15000]}  # Hard limit to prevent context blowout
        ---
        
        Return ONLY valid JSON.
        """
        
        try:
            response = await self.engine.llm.complete(
                system="You are a JSON extraction heuristic tool. Return strictly valid JSON array.",
                user=prompt
            )
            # Naive json extraction
            start = response.find('[')
            end = response.rfind(']') + 1
            if start == -1 or end == 0:
                return []
                
            parsed = json.loads(response[start:end])
            return [n for n in parsed if isinstance(n, dict) and "category" in n and "content" in n]
        except Exception as e:
            logger.error("LLM Extraction failed for %s: %s", filename, e)
            return []

    async def _seed_vault(self, notes: list[dict]) -> None:
        """Write the extracted notes into the FalkorDB Vault."""
        for note in notes:
            category = note["category"].lower()
            content = note["content"]
            
            if category not in ["identity", "preference", "constraint", "domain"]:
                continue
                
            # Add to temporal engine graph (StingerVault handles the actual node creation)
            # In v0.6, memory_add is the easiest path to inject a fact
            try:
                await self.engine.add_episode(
                    content=content,
                    source=f"agent_profile_ingester",
                    source_description=f"Auto-extracted {category}",
                    valid_at=int(asyncio.get_event_loop().time()),
                    agent_namespace=self.agent_namespace,
                )
                logger.info("Auto-seeded %s: %s", category, content[:50])
            except Exception as e:
                logger.error("Failed to seed vault note: %s", e)
