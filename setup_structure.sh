#!/bin/bash
# Setup project structure for GreenLens Hackathon

# # Clone repo
# git clone https://github.com/your-username/GreenLens.git
# cd GreenLens

# Create main folders
mkdir dataset metadata notebooks src ui docs

# Inside dataset
mkdir -p dataset/raw/snacks dataset/raw/beverages dataset/raw/dairy dataset/raw/fruits_veggies dataset/raw/staples
mkdir dataset/processed

# Metadata files
touch metadata/products.csv metadata/alternatives.csv

# Notebooks
touch notebooks/data_exploration.ipynb

# Source code folders
mkdir -p src/model src/api src/utils

# UI folder (React project will be initialized later inside ui/react-app)
mkdir ui/react-app

# Docs folder
touch docs/README.md

# Main repo readme
touch README.md
