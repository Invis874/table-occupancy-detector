@echo off
echo ========================================
echo Table Occupancy Detector
echo ========================================

REM Check for virtual environment
if not exist venv\Scripts\activate (
    echo [1/4] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [2/4] Activating environment...
call venv\Scripts\activate

REM Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [4/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Starting detection...
echo ========================================

REM Run the detector
python main.py --video video1.mp4

echo.
echo ========================================
echo Done! Press any key to exit...
pause