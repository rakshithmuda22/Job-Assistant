"""
AI Job Application Assistant v2 — Main FastAPI Application.

Flow: detect role type → run 4 analyses in parallel → return all results.
"""

import asyncio
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from llm_service import LLMService
from pdf_parser import extract_name_from_resume, extract_text_from_pdf

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Job Application Assistant",
    description="Analyse your resume against job descriptions using Groq AI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_llm_service: "LLMService | None" = None
MAX_PDF_SIZE_BYTES = 5 * 1024 * 1024
MIN_JOB_DESC_LENGTH = 20


def get_llm_service() -> LLMService:
    """Return the shared LLMService, init on first call."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the single-page application."""
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze_resume(
    resume_pdf: UploadFile = File(..., description="Resume PDF"),
    job_description: str = Form(..., description="Job description text"),
) -> JSONResponse:
    """
    Analyse a resume PDF against a job description.

    1. Detect role type (intern/entry/mid/senior)
    2. Run 4 LLM analyses in parallel with role context
    3. Return comprehensive results
    """
    filename = resume_pdf.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    jd = job_description.strip()
    if len(jd) < MIN_JOB_DESC_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Job description must be at least {MIN_JOB_DESC_LENGTH} chars.",
        )

    pdf_bytes = await resume_pdf.read()
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="PDF too large (max 5 MB).")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        resume_text = extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Not enough text extracted. Use a text-based PDF.",
        )

    applicant_name = extract_name_from_resume(resume_text)
    logger.info("Resume: %d chars, name=%s, file=%s", len(resume_text), applicant_name, filename)

    try:
        service = get_llm_service()

        # Step 1: detect role type (sequential — fast, ~200ms)
        role_info = await service.detect_role_type(jd)
        role_type = role_info["role_type"]
        logger.info("Role detected: %s", role_type)

        # Step 2: run all 4 analyses concurrently with role context
        (
            analysis_result,
            bullets_result,
            cover_letter_result,
            growth_result,
        ) = await asyncio.gather(
            service.comprehensive_analysis(resume_text, jd, role_type),
            service.rewrite_bullets_and_fixes(resume_text, jd, role_type),
            service.generate_cover_letter(resume_text, jd, applicant_name, role_type),
            service.skills_and_growth_analysis(resume_text, jd, role_type),
        )
    except RuntimeError as exc:
        logger.error("LLM analysis failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="AI service temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred.",
        ) from exc

    score = analysis_result.get("match", {}).get("overall_score", 0)
    logger.info("Analysis done — name=%s, score=%s, role=%s", applicant_name, score, role_type)

    return JSONResponse({
        "role_type": role_info,
        "match": analysis_result.get("match", {}),
        "ats_analysis": analysis_result.get("ats_analysis", {}),
        "confidence": analysis_result.get("confidence", {}),
        "strengths": analysis_result.get("strengths", {}),
        "bullets": bullets_result.get("bullets", []),
        "resume_fixes": bullets_result.get("resume_fixes", []),
        "cover_letter": cover_letter_result,
        "skills_gap": growth_result.get("skills_gap", {}),
        "action_plan": growth_result.get("action_plan", []),
        "project_suggestions": growth_result.get("project_suggestions", []),
        "applicant_name": applicant_name,
    })


@app.get("/health")
async def health_check() -> dict:
    """Health-check endpoint."""
    return {"status": "ok"}
