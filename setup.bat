@echo off
echo Keystroke Authentication System - Setup
echo =====================================
echo.

echo Step 1: Setting up backend environment...
cd backend
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
) else (
    echo .env file already exists
)

echo.
echo Step 2: Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Step 3: Setting up frontend environment...
cd ..\frontend
if not exist .env (
    copy .env.example .env
    echo Created .env file from template
) else (
    echo .env file already exists
)

echo.
echo Step 4: Installing Node.js dependencies...
npm install --legacy-peer-deps

echo.
echo Setup complete!
echo.
echo To start the system:
echo 1. Run start_backend.bat to start the API server
echo 2. Run start_frontend.bat to start the web interface
echo 3. Open http://localhost:3000 in your browser
echo.
pause
