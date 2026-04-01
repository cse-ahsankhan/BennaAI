#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# ── .env ────────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[setup] Created .env from .env.example — edit it before re-running if you need Claude API."
fi

# ── Virtual environment ──────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating virtual environment …"
    python -m venv "$VENV_DIR"
fi

# Activate
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows (Git Bash / MSYS2)
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

# ── Dependencies ─────────────────────────────────────────────────────────────
echo "[setup] Installing / verifying dependencies …"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "  ██████╗ ███████╗███╗   ██╗███╗   ██╗ █████╗      █████╗ ██╗"
echo "  ██╔══██╗██╔════╝████╗  ██║████╗  ██║██╔══██╗    ██╔══██╗██║"
echo "  ██████╔╝█████╗  ██╔██╗ ██║██╔██╗ ██║███████║    ███████║██║"
echo "  ██╔══██╗██╔══╝  ██║╚██╗██║██║╚██╗██║██╔══██║    ██╔══██║██║"
echo "  ██████╔╝███████╗██║ ╚████║██║ ╚████║██║  ██║    ██║  ██║██║"
echo "  ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝"
echo ""
echo "  Construction Document Intelligence · GCC"
echo "  Opening at http://localhost:8501"
echo ""

streamlit run app/streamlit_app.py \
    --server.headless false \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none
