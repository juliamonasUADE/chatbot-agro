print("inicio")
import os
print("os ok")
import re
print("re ok")
import json
print("json ok")
import pdfplumber
print("pdfplumber ok")
import docx
print("docx ok")
import chromadb
print("chromadb ok")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print("modelo ok")