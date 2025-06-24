import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from tqdm.auto import tqdm
import joblib
import sys
import os

print("Python executable in use:", sys.executable)
tqdm.pandas()

# Verificar si CUDA está disponible y mostrar si se está usando GPU o CPU en este caso una RTX3050
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA disponible. Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")  # Esto es para verificar que se está utilizando la CPU del sistema
    print("CUDA no disponible. Usando CPU.")

print("Iniciando carga de datos...")
ruta_archivo = r"C:\DADES\DOCLIB\CIE10\ENTRENAMIENTO\Entrenamiento.csv"
df = pd.read_csv(ruta_archivo, sep="|")
print("Columnas del DataFrame:", df.columns.tolist())
print(f"Datos cargados correctamente. Total de registros: {len(df)}")

print("Normalizando texto del criterio médico...")
df["JUICIO_CLINICO_ENRIQUECIDO"] = df["JUICIO_CLINICO_ENRIQUECIDO"].astype(str)
df["texto_procesado"] = df["JUICIO_CLINICO_ENRIQUECIDO"].str.lower().str.replace(r"[^a-zA-Z0-9ñáéíóúüç\s]", "", regex=True)

print("Codificando variable objetivo...")
df["CIE_ASIGNADO"] = df["CIE_ASIGNADO"].astype(str)
df = df.dropna(subset=["CIE_ASIGNADO", "texto_procesado"])
label_encoder = LabelEncoder()
df["CIE_ASIGNADO_COD"] = label_encoder.fit_transform(df["CIE_ASIGNADO"])

print("Validación de etiquetas antes del Dataset:")
num_labels = len(label_encoder.classes_)
max_label = df["CIE_ASIGNADO_COD"].max()
min_label = df["CIE_ASIGNADO_COD"].min()
print(f" - Clases únicas: {num_labels}")
print(f" - Máxima etiqueta codificada: {max_label}")
print(f" - Mínima etiqueta codificada: {min_label}")
assert max_label < num_labels, f"ERROR: Etiqueta {max_label} fuera de rango ({num_labels - 1})"
assert min_label >= 0, "ERROR: Hay etiquetas negativas"

print("Preparando tokenizador BERT...")
MODEL_PATH = 'C:\\DADES\\DOCLIB\\CIE10\\modelos\\bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

def tokenize_function(examples):
    return tokenizer(
        examples["texto_procesado"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

print("Tokenizando datos...")
dataset = Dataset.from_pandas(df)
dataset = dataset.rename_column("CIE_ASIGNADO_COD", "labels")
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print("Tokenización completada.")
print("Revisión rápida de labels tokenizados:", dataset[:5]["labels"])

print("Dividiendo datos en entrenamiento y prueba...")
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)}, Test: {len(eval_dataset)}")

print("Cargando modelo BERT base...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./cie10_model",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="no",  # no guardar checkpoints
    logging_strategy="epoch"  # opcional: logs solo por época
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

print("Entrenando modelo...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
print("Entrenamiento finalizado.")

eval_results = trainer.evaluate()
print(f"Resultados de evaluación final: {eval_results}")

print("Guardando modelo...")
model.save_pretrained('./cie10_model')
tokenizer.save_pretrained('./cie10_model')
joblib.dump(label_encoder, './cie10_model/label_encoder.pkl')
print("Todo listo. Modelo guardado en './cie10_model'.")
