"""
Prompt templates for AI Job Application Assistant.

Architecture (5 LLM calls — 1 sequential, then 4 parallel):
  1. detect_role_type          → classify intern/entry/mid/senior
  2. comprehensive_analysis    → structured score + ATS + confidence + strengths
  3. bullets_and_fixes         → grounded rewrites + resume-fix suggestions
  4. cover_letter              → role-aware, no overclaiming
  5. skills_and_growth         → prioritised gap + action plan + project ideas

Every prompt enforces:
  - NO hallucinated metrics — only use numbers present in the resume
  - Role-level calibration  — interns scored on potential, not tool mastery
  - JSON-only output        — raw response must be parseable by json.loads()
"""

# ──────────────────────────────────────────────────────────────────────
# Shared system prompt
# ──────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a senior technical recruiter and career coach with 15+ years "
    "at FAANG companies and top startups.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Respond with ONLY valid JSON — no markdown fences, no preamble, "
    "no commentary outside the JSON object.\n"
    "2. NEVER invent, fabricate, or hallucinate metrics, percentages, or "
    "numbers that are NOT EXPLICITLY stated in the resume.  If the resume "
    "says 'improved performance' with no number, write qualitative impact "
    "language — NEVER make up '40 %'.\n"
    "3. Calibrate expectations to the ROLE LEVEL.  An intern is evaluated "
    "on potential, learning ability, coursework, and projects — NOT on "
    "mastery of production tooling."
)


def get_role_detection_prompt(
    job_description: str,
) -> tuple[str, str]:
    """
    Classify the seniority level of a job posting.

    Args:
        job_description: Raw job-posting text.

    Returns:
        (system_prompt, user_prompt)
    """
    user = f"""Classify the seniority level of this job posting.

=== JOB DESCRIPTION ===
{job_description[:2000]}

Respond with EXACTLY this JSON:
{{
  "role_type": "<one of: intern, entry, mid, senior>",
  "reasoning": "<1 sentence justification>"
}}

Rules:
  intern — internship, co-op, or explicitly says "intern"
  entry  — 0-2 years experience, associate, new grad, junior
  mid    — 2-5 years experience, no "senior"/"lead" in title
  senior — 5+ years, senior/staff/lead/principal/architect"""
    return _SYSTEM, user


def get_comprehensive_analysis_prompt(
    resume_text: str,
    job_description: str,
    role_type: str,
) -> tuple[str, str]:
    """
    Structured match scoring, ATS analysis, confidence, strengths.

    Args:
        resume_text: Extracted resume text.
        job_description: Raw job-posting text.
        role_type: One of intern/entry/mid/senior.

    Returns:
        (system_prompt, user_prompt)
    """
    cal = ""
    if role_type == "intern":
        cal = (
            "\n\nINTERN CALIBRATION:\n"
            "  - Reduce penalty for missing production tools\n"
            "  - Increase weight for coursework, projects, learning\n"
            "  - A good intern score is 50-65 without industry tools"
        )
    elif role_type == "entry":
        cal = (
            "\n\nENTRY-LEVEL CALIBRATION:\n"
            "  - Moderate penalty for missing advanced tools\n"
            "  - Value projects, coursework, 0-2 years experience"
        )

    user = f"""Analyse this resume vs the job description.

ROLE LEVEL: {role_type}{cal}

=== RESUME (first 3000 chars) ===
{resume_text[:3000]}

=== JOB DESCRIPTION (first 2000 chars) ===
{job_description[:2000]}

Respond with EXACTLY this JSON:
{{
  "match": {{
    "overall_score": <int 0-100>,
    "breakdown": {{
      "skills_match": <int 0-100>,
      "project_relevance": <int 0-100>,
      "tools_match": <int 0-100>,
      "coursework_education": <int 0-100>
    }},
    "reasoning": "<3-4 sentences citing SPECIFIC resume items>"
  }},
  "ats_analysis": {{
    "coverage_percent": <int 0-100>,
    "matched_keywords": ["<keyword from JD found in resume>"],
    "missing_keywords": ["<keyword from JD NOT in resume>"],
    "critical_missing": ["<top 3-5 highest-impact missing>"]
  }},
  "confidence": {{
    "level": "<high | medium | low>",
    "reasoning": "<1 sentence>"
  }},
  "strengths": {{
    "strong_areas": [
      "<e.g. Solid CS fundamentals (DSA, systems)>"
    ],
    "highlights": [
      "<specific standout resume item>"
    ]
  }}
}}

Scoring: 80-100 Exceptional, 65-79 Strong, 50-64 Good,
35-49 Moderate, 20-34 Weak, 0-19 Poor.

STRICT SCORING RULES:
  Be strict and realistic. A score above 50 requires demonstrated
  technical skills directly matching the job. Soft skills like
  communication and teamwork alone should never push a score above 30.
  If the candidate has zero relevant technical skills for the role,
  the score must be in the 0-25 range regardless of soft skills.

ATS KEYWORD RULES:
  Prioritize technical skills, tools, platforms, and methodologies
  when matching keywords. Soft skills like 'teamwork',
  'communication', and 'problem solving' should NOT count toward
  the keyword coverage percentage. Only count hard/technical keyword
  matches when calculating coverage_percent.

Do NOT invent metrics not in the resume."""
    return _SYSTEM, user


def get_bullets_and_fixes_prompt(
    resume_text: str,
    job_description: str,
    role_type: str,
) -> tuple[str, str]:
    """
    Grounded bullet rewrites and resume fix suggestions.

    Args:
        resume_text: Extracted resume text.
        job_description: Raw job-posting text.
        role_type: One of intern/entry/mid/senior.

    Returns:
        (system_prompt, user_prompt)
    """
    user = f"""Rewrite resume bullets and suggest additions.

ROLE LEVEL: {role_type}

=== RESUME (first 3000 chars) ===
{resume_text[:3000]}

=== JOB DESCRIPTION (first 2000 chars) ===
{job_description[:2000]}

Respond with EXACTLY this JSON:
{{
  "bullets": [
    "• <rewritten bullet from ACTUAL resume content>"
  ],
  "resume_fixes": [
    "<specific line to ADD to the resume>"
  ]
}}

BULLET RULES (MANDATORY):
  1. Each bullet MUST trace to a REAL experience in the resume
  2. NEVER fabricate metrics — use qualitative language if no
     number exists ('Improved performance' NOT 'Improved by 40%')
  3. STAR method: Action verb + what + technology + REAL outcome
  4. Incorporate JD keywords only WHERE TRUTHFUL
  5. 6-8 bullets, each under 25 words
  6. Start each with a bullet then action verb

RESUME FIX RULES:
  1. Suggest 3-5 specific lines or sections to add
  2. Include the exact wording they should use
  3. Focus on closing biggest gaps from the JD"""
    return _SYSTEM, user


def get_cover_letter_prompt(
    resume_text: str,
    job_description: str,
    name: str,
    role_type: str,
) -> tuple[str, str]:
    """
    Role-aware, non-overclaiming cover letter.

    Args:
        resume_text: Extracted resume text.
        job_description: Raw job-posting text.
        name: Applicant name for sign-off.
        role_type: One of intern/entry/mid/senior.

    Returns:
        (system_prompt, user_prompt)
    """
    tone = ""
    if role_type == "intern":
        tone = (
            "\n\nTONE FOR INTERN ROLE:\n"
            "  - Emphasise learning mindset and curiosity\n"
            "  - Highlight transferable skills from projects\n"
            "  - NEVER imply production experience they lack\n"
            "  - Use 'eager to learn' NOT 'expert in'"
        )
    elif role_type == "entry":
        tone = (
            "\n\nTONE FOR ENTRY-LEVEL:\n"
            "  - Balance confidence with honesty\n"
            "  - Highlight projects and early experience"
        )

    user = f"""Write a personalised cover letter.

APPLICANT NAME: {name}
ROLE LEVEL: {role_type}{tone}

=== RESUME (first 2500 chars) ===
{resume_text[:2500]}

=== JOB DESCRIPTION (first 2000 chars) ===
{job_description[:2000]}

Respond with EXACTLY this JSON:
{{
  "cover_letter": "<full cover letter with \\n for paragraph breaks>"
}}

RULES:
  1. 3-4 paragraphs, 300-400 words
  2. Opening: specific hook (no 'I am writing to apply')
  3. Body: 2-3 REAL experiences mapped to job requirements
  4. Show enthusiasm for THIS company
  5. Close with call to action, sign off with name
  6. NEVER claim experience they do not have
  7. NEVER invent metrics or achievements"""
    return _SYSTEM, user


def get_skills_and_growth_prompt(
    resume_text: str,
    job_description: str,
    role_type: str,
) -> tuple[str, str]:
    """
    Prioritised skills gap, weekly action plan, project suggestions.

    Args:
        resume_text: Extracted resume text.
        job_description: Raw job-posting text.
        role_type: One of intern/entry/mid/senior.

    Returns:
        (system_prompt, user_prompt)
    """
    pri = ""
    if role_type in ("intern", "entry"):
        pri = (
            "\n\nPRIORITY CALIBRATION FOR INTERN/ENTRY:\n"
            "  - Production tools (Splunk, Datadog) -> low priority\n"
            "  - Foundational skills (data analysis) -> high priority\n"
            "  - Certifications -> low unless specifically required"
        )

    user = f"""Analyse skills gaps and create a growth plan.

ROLE LEVEL: {role_type}{pri}

=== RESUME (first 3000 chars) ===
{resume_text[:3000]}

=== JOB DESCRIPTION (first 2000 chars) ===
{job_description[:2000]}

Respond with EXACTLY this JSON:
{{
  "skills_gap": {{
    "high_priority": [
      {{"skill": "<name>", "impact": "<e.g. +8 pts>"}}
    ],
    "medium_priority": [
      {{"skill": "<name>", "impact": "<e.g. +5 pts>"}}
    ],
    "low_priority": [
      {{"skill": "<name>", "impact": "<e.g. +2 pts>"}}
    ]
  }},
  "action_plan": [
    {{
      "week": 1,
      "title": "<theme>",
      "tasks": [
        "<task with FREE resource>"
      ]
    }}
  ],
  "project_suggestions": [
    {{
      "name": "<project name>",
      "description": "<1-2 sentences>",
      "skills_covered": ["<skill>"],
      "estimated_time": "<e.g. 1-2 weeks>"
    }}
  ]
}}

RULES:
  1. skills_gap: 3-5 high, 2-4 medium, 1-3 low priority items
  2. action_plan: exactly 4 weeks, 2-3 tasks per week
  3. Each task MUST name a FREE resource
  4. project_suggestions: 2-3 portfolio-worthy projects
  5. Projects address top skill gaps, completable in 1-2 weeks
  6. Impact estimates rough but reasonable
  7. Only suggest skills explicitly mentioned in the job description
     or directly relevant to the role. Do NOT suggest tools from
     unrelated industries (e.g. no DevOps monitoring tools for an
     analytics role, no ML tools for a frontend role)."""
    return _SYSTEM, user
