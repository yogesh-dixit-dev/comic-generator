@echo off
echo Setting up Comic Generator Environment...

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete! To start working, run:
echo venv\Scripts\activate
pause
