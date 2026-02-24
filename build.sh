#!/usr/bin/env bash
# Build script for Render deployment

set -o errexit  # exit on error

pip install --upgrade pip
pip install -r requirements.txt

# Initialize the database
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database initialized successfully!')
"
