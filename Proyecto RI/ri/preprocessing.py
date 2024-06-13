import os
import re
from nltk.stem import PorterStemmer

# Definir la ruta del corpus y las stopwords en tu sistema local
REUTERS_PATH = "reuters"
STOPWORDS_PATH = os.path.join(REUTERS_PATH, "stopwords.txt")
TRAINING_PATH = os.path.join(REUTERS_PATH, "training")
PROCESSED_PATH = os.path.join(REUTERS_PATH, "processed")

# Leer las stopwords desde el archivo
with open(STOPWORDS_PATH, 'r', encoding='ascii') as file:
    stop_words = set(word.strip() for word in file.readlines())

# Función de preprocesamiento
def lmp(texto):
    # Normalización
    cleaned_text = re.sub(r'[^\w\s]', '', texto)
    cleaned_text = cleaned_text.lower()
    words = cleaned_text.split()
    # Steaming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    # Eliminar stopwords
    filtered_words = [word for word in stemmed_words if word not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

# Crear la carpeta si no existe para guardar los documentos preprocesados
if not os.path.exists(PROCESSED_PATH):
    os.makedirs(PROCESSED_PATH)

# Preprocesar y guardar documentos
for filename in os.listdir(TRAINING_PATH):
    input_filepath = os.path.join(TRAINING_PATH, filename)
    output_filepath = os.path.join(PROCESSED_PATH, filename)

    with open(input_filepath, 'r', encoding='ascii') as input_file:
        text = input_file.read()
        processed_text = lmp(text)

    with open(output_filepath, 'w', encoding='utf-8') as output_file:
        output_file.write(processed_text)

print("Documentos preprocesados y guardados en la carpeta 'processed'.")
