# QuantEdge

QuantEdge now runs with the Streamlit UI and backend code in place. The separate React frontend has been removed.

## Quick Start

### Windows
```bat
cd QuantEdge-main
setup.bat
start_streamlit.bat
```

### Linux / macOS
```bash
cd QuantEdge-main
bash setup.sh
source venv/bin/activate
python -m streamlit run app/main.py
```

### Manual Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
mkdir -p data/cache data/exports  # Windows: mkdir data\cache data\exports
python -m streamlit run app/main.py
```

If you have multiple Python versions installed, avoid plain `streamlit run app/main.py` on Windows. Use `start_streamlit.bat` or `python -m streamlit run app/main.py` from the activated project environment so Streamlit uses the same interpreter where the dependencies were installed.

## Backend API

The FastAPI backend is still available:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

On Windows you can also run:

```bat
start_api.bat
```

## Demo Mode

`DEMO_MODE=true` is enabled by default. The app can run with synthetic data and without external API keys. Set `DEMO_MODE=false` in `.env` to switch to live data.

## Running Tests

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
python -m pytest tests/ -v
```

## Project Structure

```text
QuantEdge-main/
|-- api/
|   `-- server.py
|-- app/
|   |-- main.py
|   |-- data_engine.py
|   `-- pages/
|-- core/
|-- tests/
|-- utils/
|-- start_api.bat
|-- start_streamlit.bat
|-- setup.bat
|-- setup.sh
`-- requirements.txt
```

## Notes

- Streamlit pages are restored under `app/pages/`.
- Backend logic remains in `core/`, `api/`, and `utils/`.
- The separate `frontend/` app is intentionally not restored.
