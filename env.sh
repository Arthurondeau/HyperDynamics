#!/bin/bash

# Define the environment name
ENV_NAME="HyperDM"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

# Create the virtual environment
echo "Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME

# Activate the virtual environment
echo "Activating virtual environment: $ENV_NAME"
source $ENV_NAME/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment $ENV_NAME setup complete."