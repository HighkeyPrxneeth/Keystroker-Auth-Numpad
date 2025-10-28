@echo off
echo Starting Backend Server...
echo.
cd /d "%~dp0backend"
echo Current directory: %CD%
echo.
echo Starting uvicorn server...
echo Press Ctrl+C to stop the server
echo.
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
echo.
echo Server has stopped.
pause
