import chromadb
from sentence_transformers import SentenceTransformer

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
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


def tiene_palabras_reales(pregunta):
    palabras_comunes = {
        "que", "qué", "como", "cómo", "cuanto", "cuánto", "cuántas", "cuantas",
        "donde", "dónde", "hay", "son", "tiene", "es", "de", "en", "el", "la",
        "los", "las", "un", "una", "se", "con", "por", "para", "del", "al",
        "me", "te", "le", "nos", "si", "no", "ya", "más", "mas", "muy",
        "pero", "porque", "porqué", "cuando", "cuándo", "quien", "quién",
        "cual", "cuál", "cuales", "cuáles", "sobre", "entre", "hasta", "desde",
        "this", "hay", "ser", "estar", "tener", "hacer", "poder", "saber",
        "quiero", "quiero", "puedo", "puede", "tienen", "tengo", "tenés",
        "decime", "contame", "explicame", "dime", "dame", "cuéntame",
        "partido", "general", "campo", "zona", "region", "región", "área",
        "año", "años", "mes", "meses", "dia", "día", "hoy", "ayer",
        "mucho", "poco", "todo", "todos", "nada", "algo", "algun", "algún",
        "primer", "primero", "segundo", "tercer", "último", "ultimo",
        "grande", "pequeño", "mayor", "menor", "mejor", "peor",
        "nuevo", "viejo", "antiguo", "reciente", "actual",
        "precio", "costo", "valor", "cantidad", "numero", "número",
        "tipo", "tipos", "clase", "forma", "manera", "modo",
        "información", "informacion", "dato", "datos", "detalle", "detalles",
        "maiz", "maíz", "trigo", "soja", "girasol", "sorgo",
"hectarea", "hectárea", "hectareas", "hectáreas",
"campaña", "campana", "cultivo", "cultivos", "ganado",
"produccion", "producción", "siembra", "cosecha",
"rinde", "rendimiento", "apicultura", "ganaderia",
"ganadería", "empresa", "empresas", "inversor",
"madariaga", "partido", "agro", "campo",
    }
    palabras = set(pregunta.lower().split())
    return bool(palabras & palabras_comunes)

def responder(pregunta):
    if not tiene_palabras_reales(pregunta):
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
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    print("=== DEBUG ===")
    for chunk, meta, dist in zip(resultados["documents"][0], resultados["metadatas"][0], resultados["distances"][0]):
        print(f"dist={dist:.4f} | tipo={meta.get('tipo')} | texto={chunk[:80]}")
    print("=============")

    chunks     = resultados["documents"][0]
    metadatas  = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    # Buscar primero en el dataset
    for chunk, meta, dist in zip(chunks, metadatas, distancias):
        if meta.get("tipo") == "dataset" and dist < 1.2:
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