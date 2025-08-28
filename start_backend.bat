@echo off
cd /d "c:\Users\nayus\Desktop\keystrokes\project\backend"
echo Starting backend server...
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
pause
