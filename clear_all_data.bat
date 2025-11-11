@echo off
echo ========================================
echo CLEAR ALL USERS AND MODELS
echo ========================================
echo.
echo This will:
echo   1. Delete all users from the database
echo   2. Remove all trained model files
echo.

cd backend
python clear_users.py

echo.
echo Removing model files...
if exist "models\*.joblib" (
    del /Q "models\*.joblib"
    echo ✓ Model files deleted
) else (
    echo No model files found
)

echo.
echo ========================================
echo CLEANUP COMPLETE
echo ========================================
pause
