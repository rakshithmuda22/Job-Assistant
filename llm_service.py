"""
LLM Service for AI Job Application Assistant.

5-call architecture:
  1. detect_role_type           → quick role classification
  2. comprehensive_analysis     → score breakdown + ATS + confidence + strengths
  3. rewrite_bullets_and_fixes  → grounded bullets + resume-fix suggestions
  4. generate_cover_letter      → role-aware cover letter
  5. skills_and_growth_analysis → prioritised gap + action plan + projects
"""

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq

from prompts import (
    get_bullets_and_fixes_prompt,
    get_comprehensive_analysis_prompt,
    get_cover_letter_prompt,
    get_role_detection_prompt,
    get_skills_and_growth_prompt,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "llama-3.1-8b-instant"
MAX_RETRIES: int = 2
DEFAULT_MAX_TOKENS: int = 2048
RETRY_BACKOFF_SECONDS: float = 1.5
VALID_ROLE_TYPES: set[str] = {"intern", "entry", "mid", "senior"}


class LLMService:
    """Service class for all Groq LLM interactions."""

    def __init__(self) -> None:
        """Initialise the async Groq client from GROQ_API_KEY."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Copy .env.example to .env and add your key."
            )
        self.client = AsyncGroq(api_key=api_key)
        logger.info("LLMService initialised — model: %s", MODEL_NAME)

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Call Groq with exponential-backoff retry.

        Args:
            system_prompt: System role instruction.
            user_prompt: User message.
            max_tokens: Response length cap.

        Returns:
            Raw text content from the LLM.

        Raises:
            RuntimeError: When all retries are exhausted.
        """
        last_error = None  # type: Exception | None
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "LLM attempt %d/%d failed — retry in %.1fs: %s",
                    attempt, MAX_RETRIES + 1, wait, last_error,
                )
                await asyncio.sleep(wait)
            try:
                resp = await self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return resp.choices[0].message.content  # type: ignore
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES + 1} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_json_response(raw: str, fallback: Any) -> Any:
        """
        Robustly extract JSON from an LLM response string.

        Args:
            raw: Raw LLM response string.
            fallback: Value returned when parsing fails.

        Returns:
            Parsed Python object or *fallback*.
        """
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            if "```json" in raw:
                s = raw.index("```json") + 7
                e = raw.index("```", s)
                return json.loads(raw[s:e].strip())
            if "```" in raw:
                s = raw.index("```") + 3
                e = raw.index("```", s)
                return json.loads(raw[s:e].strip())
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            s, e = raw.find("{"), raw.rfind("}") + 1
            if s != -1 and e > s:
                return json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass
        try:
            s, e = raw.find("["), raw.rfind("]") + 1
            if s != -1 and e > s:
                return json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass
        logger.warning("JSON extraction failed; returning fallback.")
        return fallback

    @staticmethod
    def _clamp(val: Any, lo: int, hi: int, default: int) -> int:
        """Coerce *val* to an int clamped between *lo* and *hi*."""
        try:
            return max(lo, min(hi, int(float(val))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_str_list(val: Any) -> list[str]:
        """Coerce a value to a list of non-empty strings."""
        if isinstance(val, list):
            return [str(i).strip() for i in val if i]
        return []

    @staticmethod
    def _ensure_skill_list(val: Any) -> list[dict[str, str]]:
        """Normalise a priority list to [{skill, impact}, ...]."""
        if not isinstance(val, list):
            return []
        out: list[dict[str, str]] = []
        for item in val:
            if isinstance(item, dict):
                out.append({
                    "skill": str(item.get("skill", "")),
                    "impact": str(item.get("impact", "")),
                })
            elif isinstance(item, str):
                out.append({"skill": item, "impact": ""})
        return out

    # ──────────────────────────────────────────────────────────────────
    # 1. Role Detection
    # ──────────────────────────────────────────────────────────────────

    async def detect_role_type(
        self, job_description: str,
    ) -> dict[str, str]:
        """
        Classify the job posting seniority level.

        Args:
            job_description: Raw job posting text.

        Returns:
            Dict with ``role_type`` and ``reasoning``.
        """
        fallback = {"role_type": "mid", "reasoning": "Could not classify."}
        sys_p, usr_p = get_role_detection_prompt(job_description)
        raw = await self._call_llm(sys_p, usr_p, max_tokens=256)
        r = self._parse_json_response(raw, fallback)
        if not isinstance(r, dict):
            return fallback
        rt = str(r.get("role_type", "mid")).lower().strip()
        if rt not in VALID_ROLE_TYPES:
            rt = "mid"
        return {
            "role_type": rt,
            "reasoning": str(r.get("reasoning", "")),
        }

    # ──────────────────────────────────────────────────────────────────
    # 2. Comprehensive Analysis
    # ──────────────────────────────────────────────────────────────────

    async def comprehensive_analysis(
        self,
        resume_text: str,
        job_description: str,
        role_type: str,
    ) -> dict[str, Any]:
        """
        Return structured score breakdown, ATS, confidence, strengths.

        Args:
            resume_text: Extracted resume text.
            job_description: Raw job posting text.
            role_type: Detected role level.

        Returns:
            Dict with match, ats_analysis, confidence, strengths.
        """
        fallback: dict[str, Any] = {
            "match": {
                "overall_score": 50,
                "breakdown": {
                    "skills_match": 50, "project_relevance": 50,
                    "tools_match": 50, "coursework_education": 50,
                },
                "reasoning": "Unable to analyse. Please try again.",
            },
            "ats_analysis": {
                "coverage_percent": 0, "matched_keywords": [],
                "missing_keywords": [], "critical_missing": [],
            },
            "confidence": {"level": "low", "reasoning": "Analysis failed."},
            "strengths": {"strong_areas": [], "highlights": []},
        }
        sys_p, usr_p = get_comprehensive_analysis_prompt(
            resume_text, job_description, role_type,
        )
        raw = await self._call_llm(sys_p, usr_p, max_tokens=2500)
        r = self._parse_json_response(raw, fallback)
        if not isinstance(r, dict):
            return fallback

        # match
        m = r.get("match", {}) if isinstance(r.get("match"), dict) else {}
        bd = m.get("breakdown", {}) if isinstance(m.get("breakdown"), dict) else {}
        parsed_match = {
            "overall_score": self._clamp(m.get("overall_score"), 0, 100, 50),
            "breakdown": {
                "skills_match": self._clamp(bd.get("skills_match"), 0, 100, 50),
                "project_relevance": self._clamp(bd.get("project_relevance"), 0, 100, 50),
                "tools_match": self._clamp(bd.get("tools_match"), 0, 100, 50),
                "coursework_education": self._clamp(
                    bd.get("coursework_education"), 0, 100, 50,
                ),
            },
            "reasoning": str(m.get("reasoning", fallback["match"]["reasoning"])),
        }

        # ats
        a = r.get("ats_analysis", {}) if isinstance(r.get("ats_analysis"), dict) else {}
        parsed_ats = {
            "coverage_percent": self._clamp(a.get("coverage_percent"), 0, 100, 0),
            "matched_keywords": self._to_str_list(a.get("matched_keywords")),
            "missing_keywords": self._to_str_list(a.get("missing_keywords")),
            "critical_missing": self._to_str_list(a.get("critical_missing")),
        }

        # confidence
        c = r.get("confidence", {}) if isinstance(r.get("confidence"), dict) else {}
        lvl = str(c.get("level", "low")).lower()
        if lvl not in ("high", "medium", "low"):
            lvl = "low"
        parsed_conf = {"level": lvl, "reasoning": str(c.get("reasoning", ""))}

        # strengths
        s = r.get("strengths", {}) if isinstance(r.get("strengths"), dict) else {}
        parsed_str = {
            "strong_areas": self._to_str_list(s.get("strong_areas")),
            "highlights": self._to_str_list(s.get("highlights")),
        }

        return {
            "match": parsed_match,
            "ats_analysis": parsed_ats,
            "confidence": parsed_conf,
            "strengths": parsed_str,
        }

    # ──────────────────────────────────────────────────────────────────
    # 3. Bullets + Resume Fixes
    # ──────────────────────────────────────────────────────────────────

    async def rewrite_bullets_and_fixes(
        self,
        resume_text: str,
        job_description: str,
        role_type: str,
    ) -> dict[str, list[str]]:
        """
        Grounded bullet rewrites and resume fix suggestions.

        Args:
            resume_text: Extracted resume text.
            job_description: Raw job posting text.
            role_type: Detected role level.

        Returns:
            Dict with ``bullets`` and ``resume_fixes``.
        """
        fallback: dict[str, list[str]] = {
            "bullets": ["Unable to generate bullets."],
            "resume_fixes": [],
        }
        sys_p, usr_p = get_bullets_and_fixes_prompt(
            resume_text, job_description, role_type,
        )
        raw = await self._call_llm(sys_p, usr_p, max_tokens=1800)
        r = self._parse_json_response(raw, None)
        if isinstance(r, dict):
            braw = r.get("bullets") or r.get("bullet_points") or []
            bullets = self._to_str_list(braw) or fallback["bullets"]
            fixes = self._to_str_list(r.get("resume_fixes", []))
            return {"bullets": bullets, "resume_fixes": fixes}
        if isinstance(r, list):
            return {
                "bullets": self._to_str_list(r) or fallback["bullets"],
                "resume_fixes": [],
            }
        return fallback

    # ──────────────────────────────────────────────────────────────────
    # 4. Cover Letter
    # ──────────────────────────────────────────────────────────────────

    async def generate_cover_letter(
        self,
        resume_text: str,
        job_description: str,
        name: str,
        role_type: str,
    ) -> str:
        """
        Write a role-aware, non-overclaiming cover letter.

        Args:
            resume_text: Extracted resume text.
            job_description: Raw job posting text.
            name: Applicant name for sign-off.
            role_type: Detected role level.

        Returns:
            Cover letter as a plain string.
        """
        sys_p, usr_p = get_cover_letter_prompt(
            resume_text, job_description, name, role_type,
        )
        raw = await self._call_llm(sys_p, usr_p, max_tokens=1500)
        r = self._parse_json_response(raw, None)
        if isinstance(r, dict):
            letter = r.get("cover_letter") or r.get("letter", "")
            if letter:
                return str(letter).strip()
        if isinstance(raw, str) and len(raw.strip()) > 80:
            return raw.strip()
        return "Unable to generate a cover letter. Please try again."

    # ──────────────────────────────────────────────────────────────────
    # 5. Skills + Growth
    # ──────────────────────────────────────────────────────────────────

    async def skills_and_growth_analysis(
        self,
        resume_text: str,
        job_description: str,
        role_type: str,
    ) -> dict[str, Any]:
        """
        Prioritised skills gap, action plan, project suggestions.

        Args:
            resume_text: Extracted resume text.
            job_description: Raw job posting text.
            role_type: Detected role level.

        Returns:
            Dict with skills_gap, action_plan, project_suggestions.
        """
        fallback: dict[str, Any] = {
            "skills_gap": {
                "high_priority": [], "medium_priority": [], "low_priority": [],
            },
            "action_plan": [],
            "project_suggestions": [],
        }
        sys_p, usr_p = get_skills_and_growth_prompt(
            resume_text, job_description, role_type,
        )
        raw = await self._call_llm(sys_p, usr_p, max_tokens=2500)
        r = self._parse_json_response(raw, fallback)
        if not isinstance(r, dict):
            return fallback

        sg = r.get("skills_gap", {})
        if not isinstance(sg, dict):
            sg = {}
        parsed_gap = {
            "high_priority": self._ensure_skill_list(sg.get("high_priority")),
            "medium_priority": self._ensure_skill_list(sg.get("medium_priority")),
            "low_priority": self._ensure_skill_list(sg.get("low_priority")),
        }

        ap_raw = r.get("action_plan", [])
        action_plan: list[dict[str, Any]] = []
        if isinstance(ap_raw, list):
            for item in ap_raw:
                if isinstance(item, dict):
                    action_plan.append({
                        "week": self._clamp(item.get("week"), 1, 52, 1),
                        "title": str(item.get("title", "")),
                        "tasks": self._to_str_list(item.get("tasks", [])),
                    })

        ps_raw = r.get("project_suggestions", [])
        projects: list[dict[str, Any]] = []
        if isinstance(ps_raw, list):
            for item in ps_raw:
                if isinstance(item, dict):
                    projects.append({
                        "name": str(item.get("name", "")),
                        "description": str(item.get("description", "")),
                        "skills_covered": self._to_str_list(
                            item.get("skills_covered", []),
                        ),
                        "estimated_time": str(item.get("estimated_time", "")),
                    })

        return {
            "skills_gap": parsed_gap,
            "action_plan": action_plan,
            "project_suggestions": projects,
        }
