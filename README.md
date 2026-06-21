# 🌾 ChatBot Agro — General Madariaga

Asistente inteligente para consultas agropecuarias del Partido de General Madariaga, Buenos Aires, Argentina. Desarrollado como proyecto de investigación en la Universidad Argentina de la Empresa (UADE) — Proyecto A25T12.

---

## 🌐 Demo en vivo

| Componente | URL |
|---|---|
| Frontend | https://juliamonasuade.github.io/chatbot-agro |
| API Backend | https://chatbot-agro-production-c45e.up.railway.app |

---

## ¿Qué es?

Un chatbot basado en **RAG (Retrieval Augmented Generation)** que responde preguntas sobre producción agrícola, ganadería, empresas, clima y más, a partir del **Mapa Productivo del Partido de General Madariaga 2022** y documentación oficial de la región.

El sistema **no usa APIs pagas** (sin OpenAI, sin Anthropic). Todo corre con modelos de código abierto.

---

## 🏗️ Arquitectura

```
Usuario (web/mobile)
        ↓ pregunta
   FastAPI (Railway)
        ↓
   rag_engine.py
     ↙         ↘
Dataset curado   ChromaDB
(99 Q&A)        (116 chunks)
                     ↑
              ingest.py lee:
              - PDF (Mapa Productivo 2022)
              - DOCX (Info producción)
              con all-MiniLM-L6-v2
```

**Flujo de respuesta:**
1. La pregunta se convierte en un embedding vectorial
2. ChromaDB busca los fragmentos más similares por coseno
3. Si hay coincidencia en el dataset curado, se responde directamente
4. Si no, se extrae la respuesta de los chunks del PDF/DOCX

---

## 🗂️ Estructura del proyecto

```
chatbot-agro/
├── documents/                  ← PDF y DOCX fuente
│   ├── Mapa Productivo General Madariaga 2022 (1).pdf
│   └── info produccion madariaga.docx
├── ingest.py                   ← Indexa los documentos en ChromaDB
├── rag_engine.py               ← Motor RAG (búsqueda + respuesta)
├── main.py                     ← API FastAPI
├── finetune.py                 ← Fine-tuning GPT-2 (experimental)
├── dataset_agro_madariaga.json ← 99 pares pregunta-respuesta
├── index.html                  ← Frontend web
├── app.js                      ← Lógica del chat
├── style.css                   ← Estilos
├── startup.sh                  ← Script de inicio para Railway
├── Procfile                    ← Configuración de deploy
└── requirements.txt            ← Dependencias Python
```

---

## 🚀 Instalación local

### Requisitos
- Python 3.12+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/juliamonasUADE/chatbot-agro.git
cd chatbot-agro

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Indexar los documentos (una sola vez)
python ingest.py

# 5. Levantar el servidor
uvicorn main:app --reload

# 6. Abrir index.html en el navegador
```

---

## 🧠 Tecnologías

| Componente | Tecnología |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | sentence-transformers 2.7.0 — all-MiniLM-L6-v2 |
| Base vectorial | ChromaDB |
| Extracción PDF | pdfplumber |
| Extracción DOCX | python-docx |
| Fine-tuning | Hugging Face Transformers — datificate/gpt2-small-spanish |
| Frontend | HTML + CSS + JavaScript |
| Deploy backend | Railway |
| Deploy frontend | GitHub Pages |

---

## 📊 Dataset

El dataset `dataset_agro_madariaga.json` contiene **99 pares pregunta-respuesta** generados a partir de los documentos fuente, cubriendo:

- 🌽 Cultivos (maíz, trigo, girasol, soja) — rindes, épocas de siembra, evolución histórica
- 🐄 Ganadería — stock bovino, distribución por especie
- 🍯 Producciones intensivas — apicultura, porcinos, kiwi, horticultura
- 🌤️ Clima — temperaturas, precipitaciones, El Niño/La Niña
- 🏭 Industria, comercio y servicios — empresas, inversión, digitalización
- 📍 Geografía — parajes Macedo y Juancho, georreferenciación
- 💰 Oportunidades de inversión — sectores con mayor crecimiento, financiamiento

---

## 🔬 Fine-tuning (experimental)

Se realizaron dos ciclos de fine-tuning con Hugging Face Transformers:

1. **gpt2** (inglés) — texto incoherente en español
2. **datificate/gpt2-small-spanish** — texto en español pero inventa datos con pocos ejemplos

**Conclusión:** GPT-2 necesita miles de ejemplos para cambiar su comportamiento generativo. Con 99 pares, el RAG puro con dataset curado supera ampliamente al fine-tuning. El modelo entrenado quedó documentado como parte del proceso de investigación.

Para entrenar:
```bash
python finetune.py
```

---

## 📄 Fuentes

- **Mapa Productivo del Partido de General Madariaga 2022** — Facultad de Ciencias Económicas y Sociales, Universidad Nacional de Mar del Plata / Municipalidad de General Madariaga.

---

## 👩‍💻 Autora

**María Julia Monasterio** — UADE, Proyecto de Investigación A25T12  
*Desarrollo de un ChatBot para la toma de decisión en sistemas agropecuarios a partir del entrenamiento de un LLM*

---

## 📜 Licencia

MIT — libre para usar, modificar y distribuir con atribución.
