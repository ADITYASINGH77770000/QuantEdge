#!/bin/bash
# ── QuantEdge Setup Script (Linux / macOS) ────────────────────────────────────
# Run once: bash setup.sh

set -e

echo "========================================"
echo "  QuantEdge — Setup Script"
echo "========================================"

# 1. Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
echo "      Done: ./venv created"

# 2. Activate and upgrade pip
echo ""
echo "[2/5] Activating venv and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip --quiet

# 3. Install dependencies
echo ""
echo "[3/5] Installing dependencies (this may take 2-5 minutes)..."
pip install -r requirements.txt --quiet
echo "      Done: all packages installed"

# 4. Create .env from example if not present
echo ""
echo "[4/5] Setting up .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "      Created .env from .env.example"
    echo "      >> Edit .env to add your API keys (optional)"
else
    echo "      .env already exists — skipping"
fi

# 5. Create data directories
echo ""
echo "[5/5] Creating data directories..."
mkdir -p data/cache data/exports
echo "      Done"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  To run the Streamlit UI:"
echo "    source venv/bin/activate"
echo "    streamlit run app/main.py"
echo ""
echo "  To run the backend API:"
echo "    source venv/bin/activate"
echo "    uvicorn api.server:app --host 0.0.0.0 --port 8000"
echo ""
echo "  To run tests:"
echo "    source venv/bin/activate"
echo "    python -m pytest tests/ -v"
echo ""
echo "  DEMO_MODE=true by default — no API keys needed."
echo "  Edit .env to add keys and set DEMO_MODE=false for live data."
echo ""
