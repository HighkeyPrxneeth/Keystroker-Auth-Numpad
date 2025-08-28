@echo off
echo Starting Frontend Server...
echo.
cd /d "c:\Users\nayus\Desktop\keystrokes\project\frontend"
echo Current directory: %CD%
echo.
echo Checking Node.js installation...
"C:\Program Files\nodejs\node.exe" --version
if errorlevel 1 (
    echo ERROR: Node.js not found at expected location
    pause
    exit /b 1
)
echo.
echo Checking npm installation...
"C:\Program Files\nodejs\npm.cmd" --version
if errorlevel 1 (
    echo ERROR: npm not found at expected location
    pause
    exit /b 1
)
echo.
echo Starting React development server...
"C:\Program Files\nodejs\npm.cmd" start
echo.
echo If you see this message, the server has stopped.
pause
