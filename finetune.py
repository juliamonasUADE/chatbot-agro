"""
finetune.py
Fine-tuning de GPT-2 con el dataset de preguntas y respuestas sobre
el Partido de General Madariaga.
Corré este archivo UNA SOLA VEZ. Genera la carpeta modelo_finetuned/
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "./dataset_agro_madariaga.json"
MODEL_BASE    = "datificate/gpt2-small-spanish"
OUTPUT_DIR    = "./modelo_finetuned"
EPOCHS        = 5
BATCH_SIZE    = 2
MAX_LENGTH    = 512
# ───────────────────────────────────────────────────────────────────────────────


def cargar_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convertir cada par en un texto con formato instrucción/respuesta
    textos = []
    for item in data:
        texto = f"### Pregunta:\n{item['instruction']}\n\n### Respuesta:\n{item['output']}\n"
        textos.append({"text": texto})

    return Dataset.from_list(textos)


def tokenizar(ejemplos, tokenizer):
    tokens = tokenizer(
        ejemplos["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def entrenar():
    print("📂 Cargando dataset...")
    dataset = cargar_dataset()
    print(f"  → {len(dataset)} ejemplos")

    print("\n🧠 Cargando modelo base GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_BASE)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_BASE)

    print("\n⚙️  Tokenizando...")
    dataset_tokenizado = dataset.map(
        lambda x: tokenizar(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    dataset_tokenizado.set_format("torch")

    print("\n🏋️  Iniciando entrenamiento...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_steps=10,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # usa GPU si está disponible
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_tokenizado,
    )

    trainer.train()

    print(f"\n💾 Guardando modelo en {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Fine-tuning completo.")


if __name__ == "__main__":
    entrenar()