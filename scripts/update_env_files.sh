#!/bin/bash

# Check if the script is running inside a Conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "Error: No Conda environment is activated."
  echo "Please activate a Conda environment before running this script."
  exit 1
fi

# Get the current environment name
ENV_NAME=$CONDA_DEFAULT_ENV

# Generate environment.yml
echo "Creating/updating environment.yml..."
conda env export | grep -v "^prefix: " > environment.yml

# Generate requirements.txt
echo "Creating/updating requirements.txt..."
# Use pip freeze to capture pip-installed packages
pip freeze > requirements.txt

# Print success message
echo "Environment files created/updated successfully!"
echo "Environment name: $ENV_NAME"
echo "- environment.yml"
echo "- requirements.txt"
