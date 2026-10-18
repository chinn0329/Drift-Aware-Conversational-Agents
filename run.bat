@echo off
echo ================================
echo  Drift-Aware Dataset Generator
echo ================================

REM Setup venv if it doesn't exist
if not exist ".venv" (
    echo Setting up environment for first time...
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
    echo Setup done!
)

REM Check .env exists
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please create a .env file with your GROQ_API_KEY
    echo Example: GROQ_API_KEY=gsk_your_key_here
    pause
    exit /b 1
)

echo Starting generation...
.venv\Scripts\python.exe -m src.pipeline.simulate --batch-size 20

echo.
echo Done! Press any key to exit.
pause
