@echo off
REM Retrain the drift classifier on whatever is currently in data\ (Windows).
cd /d "%~dp0\.."
python -m src.classifier.data || exit /b 1
python -m src.classifier.train %* || exit /b 1
python -m src.classifier.evaluate
