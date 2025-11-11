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
if not exist "node_modules" (
    echo node_modules directory not found. Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed
        pause
        exit /b 1
    )
)
echo.
echo Clearing CI/BROWSER environment flags for interactive dev server...
set "CI="
set "BROWSER=none"
echo.
echo Starting React development server...
echo Press Ctrl+C to stop the server
echo.
call npm run start
if errorlevel 1 (
    echo.
    echo ERROR: The React development server exited unexpectedly.
    echo Check the logs above for details (warnings may be treated as errors when CI is set).
    pause
    exit /b 1
)
echo.
echo Server has stopped.
pause
