#!/bin/bash

echo "Setting up Comic Generator Environment..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! To start working, run:"
echo "source venv/bin/activate"
