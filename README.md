# AI Job Application Assistant

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?logo=render)](https://your-app.onrender.com)
[![CI](https://github.com/YOUR_USERNAME/ai-job-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ai-job-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq_API-F55036)](https://console.groq.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Upload your resume, paste a job description, and get a match score, rewritten bullet points, a tailored cover letter, and a skills gap analysis — powered by Groq's blazing-fast LLaMA 3.1 API.

![Demo](demo.gif)

---

## Features

- **Resume Match Score** — AI-computed 0-100 score with a plain-English explanation of what matches and what doesn't
- **Improved Bullet Points** — ATS-optimised, STAR-method rewrites of your resume bullets tailored to the specific role
- **Tailored Cover Letter** — A 3-4 paragraph, role-specific cover letter generated from your actual experience
- **Skills Gap Analysis** — Side-by-side "You Have / You're Missing" breakdown with a concrete, time-boxed action plan
- **Dark / Light mode** — Clean, responsive UI with a one-click theme toggle
- **Drag-and-drop PDF upload** — Supports text-based PDFs up to 5 MB
- **Concurrent LLM calls** — All four analyses run in parallel via `asyncio.gather` for minimal wait time
- **Production-ready** — Dockerfile, docker-compose, and a GitHub Actions CI/CD pipeline included

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| LLM | Groq API — `llama-3.1-8b-instant` (free tier) |
| PDF Parsing | PyPDF2 |
| Frontend | Vanilla JS + CSS (single HTML file, no build step) |
| Testing | Pytest + pytest-asyncio + unittest.mock |
| CI/CD | GitHub Actions |
| Deployment | Docker / Render |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-job-assistant.git
cd ai-job-assistant
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
docker-compose up --build
```

The app is available at **http://localhost:8080**.

---

## Running Tests

```bash
pytest tests/ -v
```

All LLM calls are mocked — no API key required for tests.

---

## Deploy to Render (3 steps)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Set the environment variable `GROQ_API_KEY` in Render's dashboard → click **Deploy**

Render auto-detects the Dockerfile. Your app will be live at `https://your-app.onrender.com` within ~2 minutes.

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
              asyncio.gather()  ← runs all 4 tasks concurrently
              ┌──────┬──────────┬────────────┬──────────────┐
              │      │          │            │              │
           Match   Bullets  Cover Letter  Skills Gap
           Score             Generator    Analyser
              │      │          │            │
              └──────┴──────────┴────────────┘
                     │
              prompts.py  ← assembles role-specific prompts
                     │
              Groq API  (llama-3.1-8b-instant)
              ~200-800 ms per call, JSON output
                     │
              llm_service.py  ← parses & validates JSON
                     │
    ◄──── JSON response with all 4 results
    │
  index.html  ← renders score ring, bullets, cover letter, skills grid
```

**Why Groq?** Groq's LPU hardware runs inference 10-20× faster than GPU-based APIs, making the multi-call design practical for real users. The free tier is generous enough for a portfolio project or a small production workload.

**Why a single HTML file?** No build step, no npm, no React — the entire frontend ships as one file served directly by FastAPI. This keeps deployment friction at zero while still delivering a polished, interactive UI.

---

## Project Structure

```
job-assistant/
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
    └── test_llm_service.py  # 20+ unit tests, all mocked
```

---

## License

MIT © 2025 — see [LICENSE](LICENSE) for details.
