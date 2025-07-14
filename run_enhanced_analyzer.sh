#!/bin/bash
# Script to run the enhanced API affinity analyzer with Python 3.12 environment

# Activate the Python 3.12 virtual environment
source api_similarity_env_py312/bin/activate

# Run the enhanced analyzer
python affinity_analyzer_claude.py "$@"