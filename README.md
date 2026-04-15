# AI Job Application Assistant

[![CI](https://github.com/rakshithmuda22/Job-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/rakshithmuda22/Job-Assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq_API-F55036)](https://console.groq.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-44_passing-brightgreen?logo=pytest&logoColor=white)](#running-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


> Upload your resume, paste a job description, and get a structured match score, ATS keyword analysis, rewritten bullet points, a tailored cover letter, skills gap breakdown, action plan, and project ideas — powered by Groq's blazing-fast LLaMA 3.1 API.

---

## Features

- **Role Detection** — Automatically classifies the job as intern/entry/mid/senior and calibrates all analysis accordingly
- **Structured Match Score** — 0-100 score with breakdown bars for skills, projects, tools, and education
- **ATS Keyword Analysis** — Technical keyword coverage percentage with matched/missing/critical-missing tags
- **Improved Bullet Points** — ATS-optimised, STAR-method rewrites grounded in your actual resume content
- **Resume Fix Suggestions** — Specific lines and sections to add to close the biggest gaps
- **Tailored Cover Letter** — Role-aware, 3-4 paragraph cover letter that never overclaims experience
- **Prioritised Skills Gap** — High/medium/low priority skills with estimated score impact
- **4-Week Action Plan** — Weekly tasks with free learning resources to close skill gaps
- **Portfolio Project Ideas** — 2-3 buildable projects that address your top skill gaps
- **Confidence Indicator** — Shows how reliable the analysis is based on available data
- **Strengths Highlighting** — Identifies your strongest areas and standout resume items
- **Dark / Light mode** — Clean, responsive UI with a one-click theme toggle
- **Drag-and-drop PDF upload** — Supports text-based PDFs up to 5 MB

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| LLM | Groq API — `llama-3.1-8b-instant` (free tier) |
| PDF Parsing | PyPDF2 |
| Frontend | Vanilla JS + CSS (single HTML file, no build step) |
| Testing | Pytest + pytest-asyncio + unittest.mock (44 tests) |
| CI/CD | GitHub Actions (tests + flake8 lint) |
| Deployment | Docker / Render |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/rakshithmuda22/Job-Assistant.git
cd Job-Assistant
```

### 2. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free — no credit card required)
3. Click **API Keys → Create API Key**
4. Copy the key

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and paste your key:
# GROQ_API_KEY=gsk_...
```

### 4. Install dependencies & run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Running with Docker

```bash
docker compose up --build
# Docker Compose v1:
# docker-compose up --build
```

The app is available at **http://localhost:8080** (host port mapped to container port 8000).

---

## Running Tests

```bash
pytest tests/ -v
```

All LLM calls are mocked — no API key required for tests.

---

## How It Works

```
User browser
    │
    │  POST /analyze  (multipart: PDF + job description text)
    ▼
FastAPI  ──► pdf_parser.py
              PyPDF2 extracts raw text from the PDF
                     │
                     ▼
              Step 1: detect_role_type()  ← sequential
                     │
                     ▼
              Step 2: asyncio.gather()  ← 4 tasks in parallel
              ┌──────────┬────────────┬──────────┬──────────────┐
              │          │            │          │              │
         Comprehensive  Bullets &  Cover     Skills &
          Analysis     Fixes      Letter    Growth Plan
              │          │            │          │
              └──────────┴────────────┴──────────┘
                     │
              prompts.py  ← role-calibrated prompts
                     │
              Groq API  (llama-3.1-8b-instant)
              ~200-800 ms per call, JSON output
                     │
              llm_service.py  ← parses & validates JSON
                     │
    ◄──── JSON response with all results
    │
  index.html  ← renders score ring, ATS tags, bullets,
                 cover letter, skills gap, action plan, projects
```

---

## Project Structure

```
Job-Assistant/
├── main.py                  # FastAPI app, CORS, routes
├── llm_service.py           # Groq client wrapper, retry logic
├── prompts.py               # Prompt templates for each task
├── pdf_parser.py            # PyPDF2 text extraction + name heuristic
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml           # Test + lint on every push
├── static/
│   └── index.html           # Full SPA — CSS, JS, HTML in one file
└── tests/
    └── test_llm_service.py  # 44 unit tests, all mocked
```

---
