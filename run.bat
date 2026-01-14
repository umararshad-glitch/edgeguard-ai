@echo off
echo Starting EdgeGuard AI...
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
pause
