@echo off
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 goto failed
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python app.py
goto done
:failed
echo Setup failed. Install Python 3.10 or newer and check your internet connection.
pause
:done
