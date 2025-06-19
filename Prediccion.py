import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Cargar modelo, tokenizer y label_encoder
print("🚀 Cargando modelo...")
model = BertForSequenceClassification.from_pretrained('./cie10_model')
tokenizer = BertTokenizer.from_pretrained('./cie10_model')
label_encoder = joblib.load('./cie10_model/label_encoder.pkl')

# Verificar entrada
if len(sys.argv) < 2:
    print("❌ Debes introducir el texto del diagnóstico entre comillas.")
    print("Ejemplo:")
    print("   python Prediccion.py \"Paciente masculino con fiebre persistente y tos\"")
    sys.exit(1)

# Obtener el texto desde los argumentos
texto = " ".join(sys.argv[1:])

# Preprocesamiento
texto_limpio = texto.lower().replace(r"[^a-zA-Z0-9\s]", "")

# Tokenización
inputs = tokenizer(texto_limpio, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

# Predicción
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    cie_asignado = label_encoder.inverse_transform([predicted_class])[0]

# Resultado
print(f"\n📝 Texto ingresado: {texto}")
print(f"🔎 CIE-10 predicho: {cie_asignado}")
print(f"📊 Confianza: {probabilities[0][predicted_class].item():.2%}")
