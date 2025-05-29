@echo off
echo Starting Virtuoso Travel Recommendation System...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is required but not installed.
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create directories if they don't exist
if not exist models mkdir models
if not exist data mkdir data
if not exist api_data mkdir api_data

REM Run the application
echo Launching Streamlit application...
streamlit run app.py

REM Deactivate virtual environment on exit
deactivate 