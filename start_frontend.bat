@echo off
echo Starting Frontend Server...
echo.
cd /d "%~dp0frontend"
echo Current directory: %CD%
echo.
echo Checking Node.js installation...
node --version
if errorlevel 1 (
    echo ERROR: Node.js not found
    pause
    exit /b 1
)
echo.
echo Checking npm installation...
npm --version
if errorlevel 1 (
    echo ERROR: npm not found
    pause
    exit /b 1
)
echo.
echo Starting React development server...
echo Press Ctrl+C to stop the server
echo.
npm start
echo.
echo Server has stopped.
pause
