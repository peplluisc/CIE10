import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import sys

modelo_path = r'C:\DADES\DOCLIB\CIE10\cie10_model'

# Cargar modelo
model = BertForSequenceClassification.from_pretrained(modelo_path)
tokenizer = BertTokenizer.from_pretrained(modelo_path)
label_encoder = joblib.load(modelo_path + '\\label_encoder.pkl')

def predict_cie(texto):
    texto_limpio = texto.lower()
    inputs = tokenizer(texto_limpio, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        cie_asignado = label_encoder.inverse_transform([predicted_class])[0]
        confianza = probabilities[0][predicted_class].item()
    return cie_asignado, confianza

if __name__ == "__main__":
    texto = " ".join(sys.argv[1:])
    cie, confianza = predict_cie(texto)
    confianza_porcentaje = confianza * 100
    salida = f"{cie}|{confianza_porcentaje:.2f}"
    print(salida.encode('cp1252', errors='replace').decode('cp1252'))

