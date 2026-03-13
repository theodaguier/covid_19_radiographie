#!/bin/bash
cd "$(dirname "$0")"
/opt/miniconda3/bin/pip install -q -r requirements.txt
/opt/miniconda3/bin/streamlit run src/streamlit/app.py
