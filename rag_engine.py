import chromadb
from sentence_transformers import SentenceTransformer

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K        = 5
# ───────────────────────────────────────────────────────────────────────────────

print("🧠 Cargando modelo de embeddings...")
_modelo_embed = SentenceTransformer(EMBED_MODEL)

print("💾 Conectando a ChromaDB...")
_client    = chromadb.PersistentClient(path=CHROMA_DIR)
_coleccion = _client.get_collection("agro_madariaga")
print(f"✅ Base lista: {_coleccion.count()} chunks indexados.")


def buscar_contexto(pregunta, top_k=TOP_K):
    embedding = _modelo_embed.encode([pregunta]).tolist()

    resultados = _coleccion.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks     = resultados["documents"][0]
    metadatas  = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    chunks_filtrados = [
        (chunk, meta["fuente"])
        for chunk, meta, dist in zip(chunks, metadatas, distancias)
        if dist < 1.5
    ]

    if not chunks_filtrados:
        return "", []

    contexto = "\n\n---\n\n".join([c for c, _ in chunks_filtrados])
    fuentes  = list(set([f for _, f in chunks_filtrados]))
    return contexto, fuentes


def extraer_respuesta(pregunta, contexto):
    palabras_clave = set(pregunta.lower().split())
    stopwords = {'cuántas', 'cuantas', 'hay', 'en', 'el', 'la', 'los', 'las',
                 'de', 'del', 'que', 'es', 'son', 'se', 'y', 'a', 'con', 'qué'}
    palabras_clave = palabras_clave - stopwords

    oraciones = [o.strip() for o in contexto.replace('\n', ' ').split('.')
                 if len(o.strip()) > 20]

    puntuadas = []
    for oracion in oraciones:
        score = sum(1 for p in palabras_clave if p in oracion.lower())
        puntuadas.append((score, oracion))

    puntuadas.sort(key=lambda x: x[0], reverse=True)
    top = [o for score, o in puntuadas[:3] if score > 0]

    return ". ".join(top) + "." if top else contexto[:500]


def es_pregunta_valida(pregunta):
    """Detecta si la pregunta tiene contenido real o es ruido."""
    # Mínimo 3 caracteres
    if len(pregunta.strip()) < 3:
        return False
    
    # Si tiene más del 60% de caracteres no alfabéticos, es ruido
    letras = sum(1 for c in pregunta if c.isalpha())
    if len(pregunta) > 0 and letras / len(pregunta) < 0.4:
        return False
    
    # Si todas las letras son mayúsculas y hay más de 5, probablemente es ruido
    if pregunta.isupper() and len(pregunta) > 5:
        return False
    
    return True

def responder(pregunta):
    if not es_pregunta_valida(pregunta):
        return {
            "respuesta": "No entendí tu consulta. Podés preguntarme sobre cultivos, ganadería, empresas, clima u oportunidades de inversión en General Madariaga.",
            "fuentes": [],
            "modo": "invalido"
        }
    # Saludos
    saludos = ["hola", "buenos días", "buenas tardes", "buenas noches", "buenas", "hey", "hi"]
    if pregunta.lower().strip().rstrip("!").rstrip("?") in saludos:
        return {
            "respuesta": "¡Hola! Soy el asistente agropecuario del Partido de General Madariaga. Podés preguntarme sobre cultivos, ganadería, empresas, clima y más. ¿En qué te puedo ayudar?",
            "fuentes": [],
            "modo": "saludo"
        }

    embedding = _modelo_embed.encode([pregunta]).tolist()

    resultados = _coleccion.query(
        query_embeddings=embedding,
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    chunks     = resultados["documents"][0]
    metadatas  = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    # Buscar primero en el dataset
    for chunk, meta, dist in zip(chunks, metadatas, distancias):
        if meta.get("tipo") == "dataset" and dist < 1.5:
            if "Respuesta:" in chunk:
                respuesta = chunk.split("Respuesta:")[-1].strip()
            else:
                respuesta = chunk
            return {
                "respuesta": respuesta,
                "fuentes": ["dataset"],
                "modo": "dataset"
            }

    # Si no hay en dataset, usar chunks de documentos
    chunks_filtrados = [
        (chunk, meta["fuente"])
        for chunk, meta, dist in zip(chunks, metadatas, distancias)
        if dist < 1.5 and meta.get("tipo") == "documento"
    ]

    if not chunks_filtrados:
        return {
            "respuesta": "No encontré información sobre ese tema en los documentos disponibles.",
            "fuentes": [],
            "modo": "sin_resultado"
        }

    contexto = "\n\n".join([c for c, _ in chunks_filtrados])
    fuentes = list(set([f for _, f in chunks_filtrados]))
    respuesta = extraer_respuesta(pregunta, contexto)

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "modo": "rag"
    }