#!/bin/bash
source /home/ubuntu/embedding/venv/bin/activate
pip install uvicorn
uvicorn main:app
