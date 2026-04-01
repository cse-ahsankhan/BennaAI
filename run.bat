@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: ── .env ────────────────────────────────────────────────────────────────────
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [setup] Created .env from .env.example
    echo         Edit it if you need Claude API support, then re-run.
)

:: ── Virtual environment ──────────────────────────────────────────────────────
if not exist ".venv\" (
    echo [setup] Creating virtual environment ...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

:: ── Dependencies ─────────────────────────────────────────────────────────────
echo [setup] Installing / verifying dependencies ...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

:: ── Launch ───────────────────────────────────────────────────────────────────
echo.
echo   BENNA AI - Construction Document Intelligence
echo   Opening at http://localhost:8501
echo.

streamlit run app/streamlit_app.py ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --server.fileWatcherType none
