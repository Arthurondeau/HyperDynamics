#!/bin/bash

# Setup script for the HyperDN environment
sudo apt install postgresql

#Start the service 
brew services start postgresql  # macOS with Homebrew

# Create database 
createdb cable_gate
# Load data into the database
psql cable_gate < datasets/cable_db/cable_db.sql

# === SAFETY CHECK: Must be sourced ===
# Define the path to the environment.yml file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML_PATH="$SCRIPT_DIR/environment_sql.yml"

# Create a new conda environment using the environment.yml file
echo "Creating the HyperDN conda environment..."
conda env create -f "$ENV_YML_PATH"

# Activate the environment
echo "Activating the SQL environment..."
conda activate "SQL"

# Verify installation of dependencies
echo "Verifying installed packages..."
#conda list

echo "Setup complete. The SQL environment is ready to use."

# Set up the PYTHONPATH to include the project directory
export SCRIPTDIR=$(dirname "$(realpath "$0")")
export BASEDIR=$(realpath "$SCRIPTDIR/..")
export PYTHONPATH="$BASEDIR:$PYTHONPATH"