#!/bin/bash
echo "Indexando documentos..."
python ingest.py
echo "Levantando servidor..."
uvicorn main:app --host 0.0.0.0 --port $PORT