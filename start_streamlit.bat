@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_BIN="
if exist ".venv\Scripts\python.exe" set "PYTHON_BIN=.venv\Scripts\python.exe"
if not defined PYTHON_BIN if exist "venv\Scripts\python.exe" set "PYTHON_BIN=venv\Scripts\python.exe"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

"%PYTHON_BIN%" -m streamlit run app/main.py
