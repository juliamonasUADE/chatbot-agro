import os
import re
import json
import pdfplumber
import docx
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "./documents"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"

def leer_pdf(path):
    paginas = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                paginas.append(t)
    texto = " ".join(paginas)
    texto = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto)
    return texto

def leer_docx(path):
    doc = docx.Document(path)
    parrafos = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parrafos)

def leer_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for item in data:
        texto = f"Pregunta: {item['instruction']}\nRespuesta: {item['output']}"
        chunks.append(texto)
    return chunks

def limpiar_texto(texto):
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r' {2,}', ' ', texto)
    return texto.strip()

def dividir_en_chunks(texto, chunk_size, overlap):
    palabras = texto.split()
    chunks = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + chunk_size
        chunk = " ".join(palabras[inicio:fin])
        chunks.append(chunk)
        inicio += chunk_size - overlap
    return chunks

def indexar_documentos():
    print("Leyendo documentos...")
    todos_chunks = []
    todos_ids = []
    todos_meta = []

    for archivo in os.listdir(DOCS_DIR):
        ruta = os.path.join(DOCS_DIR, archivo)
        if archivo.endswith(".pdf"):
            print("PDF:", archivo)
            texto = limpiar_texto(leer_pdf(ruta))
        elif archivo.endswith(".docx"):
            print("DOCX:", archivo)
            texto = limpiar_texto(leer_docx(ruta))
        else:
            continue
        chunks = dividir_en_chunks(texto, CHUNK_SIZE, CHUNK_OVERLAP)
        print("chunks:", len(chunks))
        for i, chunk in enumerate(chunks):
            todos_chunks.append(chunk)
            todos_ids.append(f"{archivo}_chunk_{i}")
            todos_meta.append({"fuente": archivo, "chunk_idx": i, "tipo": "documento"})

    dataset_path = "./dataset_agro_madariaga.json"
    if os.path.exists(dataset_path):
        print("Dataset:", dataset_path)
        chunks_dataset = leer_dataset(dataset_path)
        print("pares:", len(chunks_dataset))
        for i, chunk in enumerate(chunks_dataset):
            todos_chunks.append(chunk)
            todos_ids.append(f"dataset_chunk_{i}")
            todos_meta.append({"fuente": "dataset", "chunk_idx": i, "tipo": "dataset"})

    print("Total chunks:", len(todos_chunks))
    print("Cargando modelo...")
    modelo = SentenceTransformer(EMBED_MODEL)
    print("Generando embeddings...")
    embeddings = modelo.encode(todos_chunks, show_progress_bar=True).tolist()
    print("Guardando en ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("agro_madariaga")
    except Exception:
        pass
    coleccion = client.create_collection(name="agro_madariaga", metadata={"hnsw:space": "cosine"})
    batch = 500
    for i in range(0, len(todos_chunks), batch):
        coleccion.add(
            documents=todos_chunks[i:i+batch],
            embeddings=embeddings[i:i+batch],
            ids=todos_ids[i:i+batch],
            metadatas=todos_meta[i:i+batch],
        )
    print("Indexacion completa:", coleccion.count(), "chunks")

indexar_documentos()