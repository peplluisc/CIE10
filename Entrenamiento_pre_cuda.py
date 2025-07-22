import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from tqdm.auto import tqdm
import joblib
import sys

# Mostrar entorno de ejecución
print("Python executable in use:", sys.executable)
tqdm.pandas()

# Fase 1: Carga de datos
print("🚀 Iniciando carga de datos...")
ruta_archivo = r"C:\DADES\DOCLIB\CIE10\Entrenamiento.csv"
df = pd.read_csv(ruta_archivo, sep="|")
print("Columnas del DataFrame:", df.columns.tolist())
print(f"✅ Datos cargados correctamente. Total de registros: {len(df)}")

# Fase 2: Normalización del texto del criterio médico
print("🧹 Normalizando texto del criterio médico...")
df["JUICIO_CLINICO_ENRIQUECIDO"] = df["JUICIO_CLINICO_ENRIQUECIDO"].astype(str)
# Mantener letras acentuadas, ñ, ç
df["texto_procesado"] = df["JUICIO_CLINICO_ENRIQUECIDO"].str.lower().str.replace(r"[^a-zA-Z0-9ñáéíóúüç\s]", "", regex=True)


# Fase 3: Codificación de la variable objetivo
print("🎯 Codificando variable objetivo...")
df["CIE_ASIGNADO"] = df["CIE_ASIGNADO"].astype(str)
label_encoder = LabelEncoder()
df["CIE_ASIGNADO_COD"] = label_encoder.fit_transform(df["CIE_ASIGNADO"])

# Fase 4: Tokenización
print("🧩 Preparando tokenizador BERT...")
tokenizer = BertTokenizer.from_pretrained('C:\\DADES\\DOCLIB\\CIE10\\modelos\\bert-base-multilingual-cased')


def tokenize_function(examples):
    return tokenizer(
        examples["texto_procesado"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

print("🔄 Tokenizando datos...")
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True)
print("✅ Tokenización completada.")

# Fase 5: Preparación del dataset
dataset = dataset.rename_column("CIE_ASIGNADO_COD", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Fase 6: División en entrenamiento y evaluación
print("🧪 Dividiendo datos en entrenamiento y prueba...")
split = dataset.train_test_split(test_size=0.2)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"✅ Train: {len(train_dataset)}, Test: {len(eval_dataset)}")

# Fase 7: Definición y configuración del modelo
print("⚙️ Cargando modelo BERT base...")
# Multilingüe (español + catalán):
model = BertForSequenceClassification.from_pretrained('C:\\DADES\\DOCLIB\\CIE10\\modelos\\bert-base-multilingual-cased', num_labels=len(label_encoder.classes_))


training_args = TrainingArguments(
    output_dir="./cie10_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10)

# Fase 8: Definición de métricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Fase 9: Entrenamiento del modelo
print("🏋️‍♂️ Entrenando modelo...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics  # Añadir métricas
)

trainer.train()
print("✅ Entrenamiento finalizado.")

# Fase 10: Evaluación final
eval_results = trainer.evaluate()
print(f"📊 Resultados de evaluación final: {eval_results}")

# Fase 11: Guardado del modelo
print("💾 Guardando modelo...")
model.save_pretrained('./cie10_model')
tokenizer.save_pretrained('./cie10_model')
joblib.dump(label_encoder, './cie10_model/label_encoder.pkl')
print("🎉 Todo listo. Modelo guardado en './cie10_model'.")
