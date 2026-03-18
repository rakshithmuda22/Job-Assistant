"""
PDF parsing utilities for AI Job Application Assistant.

Provides text extraction from PDF bytes and a heuristic name extractor
that works for most standard resume layouts.
"""

import io
import logging
import re

import PyPDF2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PAGES: int = 10          # Only parse first N pages for performance
MIN_TEXT_LENGTH: int = 100   # Below this → probably a scanned / empty PDF


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all readable text from a PDF supplied as raw bytes.

    Iterates up to ``MAX_PAGES`` pages and concatenates their text.
    Handles encrypted PDFs (attempts decryption with an empty password),
    and raises helpful ``ValueError`` messages for scanned or empty PDFs.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Concatenated text content across all processed pages.

    Raises:
        ValueError: For empty input, password-protected files, scanned
            documents, or any other condition that prevents text extraction.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        logger.error("PyPDF2 failed to open PDF: %s", exc)
        raise ValueError(
            "Could not read the PDF file. It may be corrupted. "
            "Please try saving it again and re-uploading."
        ) from exc

    # ---- Encryption check --------------------------------------------------
    if reader.is_encrypted:
        logger.warning("PDF is encrypted — attempting blank-password decrypt")
        try:
            result = reader.decrypt("")
            # decrypt() returns 0 (failed), 1 (owner), or 2 (user)
            if result == 0:
                raise ValueError(
                    "This PDF is password-protected. "
                    "Please remove the password before uploading."
                )
        except Exception as exc:
            raise ValueError(
                "This PDF is password-protected. "
                "Please remove the password before uploading."
            ) from exc

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise ValueError("The PDF has no pages.")

    pages_to_read = min(num_pages, MAX_PAGES)
    logger.info("Extracting text from %d of %d page(s)", pages_to_read, num_pages)

    page_texts: list[str] = []
    for page_num in range(pages_to_read):
        try:
            page_text = reader.pages[page_num].extract_text() or ""
            if page_text.strip():
                page_texts.append(page_text)
        except Exception as exc:
            logger.warning("Skipping page %d — extraction error: %s", page_num, exc)

    full_text = "\n".join(page_texts).strip()

    if not full_text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It is likely a scanned image.  Please use a digitally-created "
            "(selectable-text) PDF, or convert it first with a tool like "
            "Adobe Acrobat or SmallPDF."
        )

    if len(full_text) < MIN_TEXT_LENGTH:
        raise ValueError(
            "Very little text was extracted from the PDF. "
            "Please make sure your resume contains selectable text."
        )

    logger.info("Extracted %d characters from PDF (%d pages)", len(full_text), pages_to_read)
    return full_text


def extract_name_from_resume(text: str) -> str:
    """
    Heuristically extract the applicant's name from resume text.

    Strategy (in priority order):
    1. First non-empty line if it looks like a proper name (1-4 title-cased words,
       letters and hyphens only).
    2. First line in the first 5 lines that looks like a proper name.
    3. First two consecutive capitalised words from the opening lines.
    4. Fallback: ``"Applicant"``.

    Args:
        text: Full text extracted from the resume PDF.

    Returns:
        Best-guess name string, or ``"Applicant"`` if none found.
    """
    if not text or not text.strip():
        return "Applicant"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "Applicant"

    def _looks_like_name(candidate: str) -> bool:
        """Return True if *candidate* resembles a person's name."""
        words = candidate.split()
        if not (1 <= len(words) <= 4):
            return False
        # Each word: letters, hyphens, apostrophes only; starts with upper-case
        return all(
            re.fullmatch(r"[A-Z][a-zA-Z'-]*", w) for w in words
        )

    # Strategy 1 & 2: scan first 5 lines
    for line in lines[:5]:
        if _looks_like_name(line):
            logger.info("Name extracted from resume lines: '%s'", line)
            return line

    # Strategy 3: first two consecutive capitalised words anywhere in top 3 lines
    for line in lines[:3]:
        cap_words = [w for w in line.split() if w and re.match(r"[A-Z][a-z]+", w)]
        if len(cap_words) >= 2:
            name = " ".join(cap_words[:2])
            logger.info("Name extracted (capitalised-word fallback): '%s'", name)
            return name

    logger.info("Could not extract name — using 'Applicant'")
    return "Applicant"
