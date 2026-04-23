@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_BIN="
if exist ".venv311\Scripts\python.exe" set "PYTHON_BIN=.venv311\Scripts\python.exe"
if not defined PYTHON_BIN if exist ".venv\Scripts\python.exe" set "PYTHON_BIN=.venv\Scripts\python.exe"
if not defined PYTHON_BIN if exist "venv\Scripts\python.exe" set "PYTHON_BIN=venv\Scripts\python.exe"
if not defined PYTHON_BIN if exist "..\.venv311\Scripts\python.exe" set "PYTHON_BIN=..\.venv311\Scripts\python.exe"
if not defined PYTHON_BIN if exist "..\.venv\Scripts\python.exe" set "PYTHON_BIN=..\.venv\Scripts\python.exe"
if not defined PYTHON_BIN if exist "..\venv\Scripts\python.exe" set "PYTHON_BIN=..\venv\Scripts\python.exe"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

echo Using Python: %PYTHON_BIN%
"%PYTHON_BIN%" -m streamlit run app/main.py
