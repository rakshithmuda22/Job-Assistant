"""
Unit tests for LLMService v2.

All Groq API calls are mocked — no real network requests are made.
Covers: JSON parsing, helper methods, all 5 analysis methods, retry logic,
and edge cases.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_service import LLMService

# ---------------------------------------------------------------------------
# Fixtures & shared test data
# ---------------------------------------------------------------------------

SAMPLE_RESUME = """
John Doe
Software Engineer | john.doe@email.com | github.com/johndoe

EXPERIENCE
Software Engineer — TechCorp (2021–2023)
• Built REST APIs using Python 3.11 and FastAPI; reduced p99 latency by 40 ms
• Migrated monolith to microservices; served 500k daily active users
• Mentored 2 junior engineers; improved sprint velocity by 18 %

EDUCATION
B.S. Computer Science — Penn State University (2021)

SKILLS
Python, JavaScript, TypeScript, React, FastAPI, PostgreSQL, Docker, AWS
""".strip()

SAMPLE_JOB = """
Senior Software Engineer — AI Platform Team

We are looking for an experienced Python engineer to help build our LLM-powered
product suite.

Requirements:
  • 3+ years of Python backend experience
  • FastAPI or Django REST Framework
  • AWS (Lambda, ECS, or EKS)
  • Experience with LLMs or ML pipelines preferred

Responsibilities:
  • Design and build scalable REST APIs
  • Collaborate with ML engineers on model serving
  • Write clean, well-tested code
""".strip()

SAMPLE_INTERN_JOB = """
Software Engineering Intern — Summer 2025

Join our engineering team for a 12-week internship.

Requirements:
  • Currently pursuing a BS/MS in Computer Science
  • Familiarity with Python or Java
  • Interest in web development

Responsibilities:
  • Contribute to team projects under mentorship
  • Learn our tech stack and development workflow
""".strip()


def _make_completion(content: str) -> MagicMock:
    """Helper: build a fake Groq completion object returning *content*."""
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture
def service() -> LLMService:
    """Return an LLMService whose Groq client is fully mocked."""
    with patch("llm_service.AsyncGroq"), \
         patch.dict("os.environ", {"GROQ_API_KEY": "test-key-ci"}):
        svc = LLMService()
        svc.client = MagicMock()
        return svc


# ---------------------------------------------------------------------------
# _parse_json_response (static method — no fixture needed)
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    """Tests for the robust JSON extraction helper."""

    def test_valid_json_object(self):
        result = LLMService._parse_json_response('{"score": 75}', {})
        assert result == {"score": 75}

    def test_valid_json_array(self):
        result = LLMService._parse_json_response('["a", "b"]', [])
        assert result == ["a", "b"]

    def test_json_in_code_fence_json(self):
        raw = '```json\n{"score": 80, "reasoning": "good"}\n```'
        result = LLMService._parse_json_response(raw, {})
        assert result["score"] == 80

    def test_json_in_plain_code_fence(self):
        raw = '```\n{"score": 60}\n```'
        result = LLMService._parse_json_response(raw, {})
        assert result["score"] == 60

    def test_json_embedded_in_prose(self):
        raw = (
            'Sure! Here is the result: '
            '{"score": 90, "reasoning": "excellent"} '
            'Hope that helps!'
        )
        result = LLMService._parse_json_response(raw, {})
        assert result["score"] == 90

    def test_malformed_returns_fallback(self):
        fallback = {"score": 50}
        result = LLMService._parse_json_response("Not JSON.", fallback)
        assert result == fallback

    def test_empty_string_returns_fallback(self):
        result = LLMService._parse_json_response("", {"error": "empty"})
        assert result == {"error": "empty"}

    def test_partial_json_returns_fallback(self):
        result = LLMService._parse_json_response(
            '{"score": 70, "reasoning":', [],
        )
        assert result == []


# ---------------------------------------------------------------------------
# _clamp helper
# ---------------------------------------------------------------------------

class TestClamp:
    """Tests for the _clamp utility."""

    def test_normal_int(self):
        assert LLMService._clamp(50, 0, 100, 0) == 50

    def test_string_number_coerced(self):
        assert LLMService._clamp("82", 0, 100, 0) == 82

    def test_float_string_coerced(self):
        assert LLMService._clamp("72.5", 0, 100, 0) == 72

    def test_above_max_clamped(self):
        assert LLMService._clamp(150, 0, 100, 50) == 100

    def test_below_min_clamped(self):
        assert LLMService._clamp(-10, 0, 100, 50) == 0

    def test_none_returns_default(self):
        assert LLMService._clamp(None, 0, 100, 50) == 50

    def test_garbage_returns_default(self):
        assert LLMService._clamp("abc", 0, 100, 50) == 50


# ---------------------------------------------------------------------------
# _ensure_skill_list helper
# ---------------------------------------------------------------------------

class TestEnsureSkillList:
    """Tests for the skill-list normaliser."""

    def test_dict_items(self):
        raw = [{"skill": "Python", "impact": "+8 pts"}]
        result = LLMService._ensure_skill_list(raw)
        assert result == [{"skill": "Python", "impact": "+8 pts"}]

    def test_string_items_wrapped(self):
        result = LLMService._ensure_skill_list(["Docker", "AWS"])
        assert result == [
            {"skill": "Docker", "impact": ""},
            {"skill": "AWS", "impact": ""},
        ]

    def test_not_a_list_returns_empty(self):
        assert LLMService._ensure_skill_list("not a list") == []

    def test_none_returns_empty(self):
        assert LLMService._ensure_skill_list(None) == []


# ---------------------------------------------------------------------------
# 1. detect_role_type
# ---------------------------------------------------------------------------

class TestDetectRoleType:
    """Tests for role-type classification."""

    async def test_detects_senior(self, service):
        payload = {
            "role_type": "senior",
            "reasoning": "Title says Senior, 3+ years.",
        }
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.detect_role_type(SAMPLE_JOB)
        assert result["role_type"] == "senior"
        assert "reasoning" in result

    async def test_detects_intern(self, service):
        payload = {
            "role_type": "intern",
            "reasoning": "Explicitly says internship.",
        }
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.detect_role_type(SAMPLE_INTERN_JOB)
        assert result["role_type"] == "intern"

    async def test_invalid_role_falls_back_to_mid(self, service):
        payload = {"role_type": "wizard", "reasoning": "Not a real level."}
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.detect_role_type(SAMPLE_JOB)
        assert result["role_type"] == "mid"

    async def test_malformed_json_returns_fallback(self, service):
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Sorry, cannot classify."),
        )
        result = await service.detect_role_type(SAMPLE_JOB)
        assert result["role_type"] == "mid"
        assert isinstance(result["reasoning"], str)


# ---------------------------------------------------------------------------
# 2. comprehensive_analysis
# ---------------------------------------------------------------------------

class TestComprehensiveAnalysis:
    """Tests for the match score, ATS, confidence, strengths analysis."""

    @staticmethod
    def _good_payload() -> dict:
        return {
            "match": {
                "overall_score": 72,
                "breakdown": {
                    "skills_match": 80,
                    "project_relevance": 65,
                    "tools_match": 70,
                    "coursework_education": 75,
                },
                "reasoning": "Strong Python skills, missing ML.",
            },
            "ats_analysis": {
                "coverage_percent": 68,
                "matched_keywords": ["Python", "FastAPI", "Docker"],
                "missing_keywords": ["Kubernetes", "LLM"],
                "critical_missing": ["LLM", "ML pipelines"],
            },
            "confidence": {
                "level": "medium",
                "reasoning": "Decent data but some gaps.",
            },
            "strengths": {
                "strong_areas": ["Backend development", "API design"],
                "highlights": ["Built REST APIs at TechCorp"],
            },
        }

    async def test_happy_path(self, service):
        payload = self._good_payload()
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.comprehensive_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "senior",
        )
        assert result["match"]["overall_score"] == 72
        assert result["match"]["breakdown"]["skills_match"] == 80
        assert "Python" in result["ats_analysis"]["matched_keywords"]
        assert result["confidence"]["level"] == "medium"
        assert len(result["strengths"]["strong_areas"]) >= 1

    async def test_score_clamped_above_100(self, service):
        payload = self._good_payload()
        payload["match"]["overall_score"] = 150
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.comprehensive_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "senior",
        )
        assert result["match"]["overall_score"] == 100

    async def test_score_clamped_below_0(self, service):
        payload = self._good_payload()
        payload["match"]["overall_score"] = -20
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.comprehensive_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "senior",
        )
        assert result["match"]["overall_score"] == 0

    async def test_invalid_confidence_level(self, service):
        payload = self._good_payload()
        payload["confidence"]["level"] = "very_high"
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.comprehensive_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "mid",
        )
        assert result["confidence"]["level"] == "low"

    async def test_malformed_json_returns_fallback(self, service):
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion("I cannot analyse this."),
        )
        result = await service.comprehensive_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "entry",
        )
        assert "match" in result
        assert "ats_analysis" in result
        assert "confidence" in result
        assert "strengths" in result
        assert 0 <= result["match"]["overall_score"] <= 100


# ---------------------------------------------------------------------------
# 3. rewrite_bullets_and_fixes
# ---------------------------------------------------------------------------

class TestRewriteBulletsAndFixes:
    """Tests for the bullet rewriter + resume fix suggestions."""

    async def test_happy_path(self, service):
        payload = {
            "bullets": [
                "• Built REST APIs with FastAPI serving 500k DAU",
                "• Mentored 2 junior engineers on best practices",
            ],
            "resume_fixes": [
                "Add a 'Machine Learning' skills section",
            ],
        }
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.rewrite_bullets_and_fixes(
            SAMPLE_RESUME, SAMPLE_JOB, "senior",
        )
        assert isinstance(result["bullets"], list)
        assert len(result["bullets"]) == 2
        assert isinstance(result["resume_fixes"], list)
        assert len(result["resume_fixes"]) == 1

    async def test_bare_array_treated_as_bullets(self, service):
        bullets = ["• Bullet A", "• Bullet B"]
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(bullets)),
        )
        result = await service.rewrite_bullets_and_fixes(
            SAMPLE_RESUME, SAMPLE_JOB, "mid",
        )
        assert len(result["bullets"]) == 2
        assert result["resume_fixes"] == []

    async def test_bullet_points_key_accepted(self, service):
        payload = {"bullet_points": ["• X", "• Y"], "resume_fixes": []}
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.rewrite_bullets_and_fixes(
            SAMPLE_RESUME, SAMPLE_JOB, "entry",
        )
        assert len(result["bullets"]) == 2

    async def test_malformed_json_returns_fallback(self, service):
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion("No bullets available."),
        )
        result = await service.rewrite_bullets_and_fixes(
            SAMPLE_RESUME, SAMPLE_JOB, "intern",
        )
        assert isinstance(result["bullets"], list)
        assert len(result["bullets"]) >= 1
        assert isinstance(result["resume_fixes"], list)

    async def test_empty_bullets_returns_fallback(self, service):
        payload = {"bullets": [], "resume_fixes": ["Add skills"]}
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.rewrite_bullets_and_fixes(
            SAMPLE_RESUME, SAMPLE_JOB, "mid",
        )
        assert len(result["bullets"]) >= 1


# ---------------------------------------------------------------------------
# 4. generate_cover_letter
# ---------------------------------------------------------------------------

class TestGenerateCoverLetter:
    """Tests for the cover-letter generator (now takes role_type)."""

    async def test_extracts_from_json(self, service):
        letter = (
            "Dear Hiring Manager,\n\n"
            "I am thrilled to apply for this role.\n\n"
            "Sincerely,\nJohn"
        )
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(
                json.dumps({"cover_letter": letter}),
            ),
        )
        result = await service.generate_cover_letter(
            SAMPLE_RESUME, SAMPLE_JOB, "John Doe", "senior",
        )
        assert isinstance(result, str)
        assert "Hiring Manager" in result

    async def test_accepts_plain_text_response(self, service):
        plain = "Dear Hiring Manager,\n\n" + "Cover letter body. " * 20
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(plain),
        )
        result = await service.generate_cover_letter(
            SAMPLE_RESUME, SAMPLE_JOB, "Jane", "entry",
        )
        assert isinstance(result, str)
        assert len(result) > 50

    async def test_letter_key_fallback(self, service):
        letter = "Dear Team,\n\nLooking forward.\n\nBest, Dev"
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(
                json.dumps({"letter": letter}),
            ),
        )
        result = await service.generate_cover_letter(
            SAMPLE_RESUME, SAMPLE_JOB, "Dev", "intern",
        )
        assert "Dear Team" in result

    async def test_short_response_returns_fallback(self, service):
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Error"),
        )
        result = await service.generate_cover_letter(
            SAMPLE_RESUME, SAMPLE_JOB, "Test", "mid",
        )
        assert "Unable to generate" in result


# ---------------------------------------------------------------------------
# 5. skills_and_growth_analysis
# ---------------------------------------------------------------------------

class TestSkillsAndGrowthAnalysis:
    """Tests for the prioritised skills gap + action plan + projects."""

    @staticmethod
    def _good_payload() -> dict:
        return {
            "skills_gap": {
                "high_priority": [
                    {"skill": "LLM APIs", "impact": "+10 pts"},
                ],
                "medium_priority": [
                    {"skill": "Kubernetes", "impact": "+5 pts"},
                ],
                "low_priority": [
                    {"skill": "Terraform", "impact": "+2 pts"},
                ],
            },
            "action_plan": [
                {
                    "week": 1,
                    "title": "LLM Fundamentals",
                    "tasks": [
                        "Complete HuggingFace NLP course (free)",
                        "Build a simple chatbot with OpenAI API",
                    ],
                },
                {
                    "week": 2,
                    "title": "Container Orchestration",
                    "tasks": ["Kubernetes basics on KodeKloud (free)"],
                },
            ],
            "project_suggestions": [
                {
                    "name": "LLM Resume Analyser",
                    "description": "Build a tool that uses LLMs.",
                    "skills_covered": ["LLM APIs", "Python"],
                    "estimated_time": "1-2 weeks",
                },
            ],
        }

    async def test_happy_path(self, service):
        payload = self._good_payload()
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.skills_and_growth_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "senior",
        )
        gap = result["skills_gap"]
        assert len(gap["high_priority"]) == 1
        assert gap["high_priority"][0]["skill"] == "LLM APIs"
        assert len(result["action_plan"]) == 2
        assert result["action_plan"][0]["week"] == 1
        assert len(result["project_suggestions"]) == 1

    async def test_string_skills_normalised(self, service):
        payload = {
            "skills_gap": {
                "high_priority": ["Docker", "AWS"],
                "medium_priority": [],
                "low_priority": [],
            },
            "action_plan": [],
            "project_suggestions": [],
        }
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.skills_and_growth_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "entry",
        )
        hp = result["skills_gap"]["high_priority"]
        assert hp[0] == {"skill": "Docker", "impact": ""}
        assert hp[1] == {"skill": "AWS", "impact": ""}

    async def test_malformed_json_returns_fallback(self, service):
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Cannot analyse skills."),
        )
        result = await service.skills_and_growth_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "intern",
        )
        assert "skills_gap" in result
        assert "action_plan" in result
        assert "project_suggestions" in result
        assert isinstance(result["skills_gap"]["high_priority"], list)

    async def test_week_clamped(self, service):
        payload = self._good_payload()
        payload["action_plan"][0]["week"] = 999
        service.client.chat.completions.create = AsyncMock(
            return_value=_make_completion(json.dumps(payload)),
        )
        result = await service.skills_and_growth_analysis(
            SAMPLE_RESUME, SAMPLE_JOB, "mid",
        )
        assert result["action_plan"][0]["week"] == 52


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Tests for the _call_llm retry/backoff mechanism."""

    async def test_succeeds_on_second_attempt(self, service):
        payload = {
            "role_type": "senior",
            "reasoning": "Retry success.",
        }
        good_completion = _make_completion(json.dumps(payload))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            service.client.chat.completions.create = AsyncMock(
                side_effect=[Exception("rate limit"), good_completion],
            )
            result = await service.detect_role_type(SAMPLE_JOB)

        assert result["role_type"] == "senior"
        assert service.client.chat.completions.create.call_count == 2

    async def test_raises_after_all_retries_exhausted(self, service):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            service.client.chat.completions.create = AsyncMock(
                side_effect=Exception("persistent error"),
            )
            with pytest.raises(RuntimeError, match="LLM call failed after"):
                await service.detect_role_type(SAMPLE_JOB)

        # MAX_RETRIES = 2 → 3 total attempts
        assert service.client.chat.completions.create.call_count == 3

    async def test_sleep_called_between_retries(self, service):
        payload = {"role_type": "entry", "reasoning": "ok"}
        good = _make_completion(json.dumps(payload))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            service.client.chat.completions.create = AsyncMock(
                side_effect=[Exception("timeout"), good],
            )
            await service.detect_role_type(SAMPLE_JOB)

        mock_sleep.assert_called_once()
