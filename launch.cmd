
@echo off
setlocal
cd /d "%~dp0"

title EcoRoute AI
color 0A

echo ==========================================
echo              ECOROUTE AI
echo      Intelligent Freight Mobility
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    pause
    exit /b 1
)

if not exist "models\final_model.joblib" (
    echo ERROR: Trained model not found.
    pause
    exit /b 1
)

echo Starting EcoRoute AI server...
echo.
echo Dashboard: http://127.0.0.1:8000/dashboard
echo Fuel Lab:  http://127.0.0.1:8000/fuel-lab
echo Analytics: http://127.0.0.1:8000/analytics
echo API Docs:  http://127.0.0.1:8000/docs
echo.

".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

pause
