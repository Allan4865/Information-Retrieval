from flask import Flask, request, render_template
import os
import re
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Definir la ruta del corpus y las stopwords
CORPUS_PATH = "reuters/training"
PROCESSED_PATH = "reuters/processed"
STOPWORDS_PATH = "reuters/stopwords"
BOW_INDEX_PATH = "reuters/bow/indice_invertido_bow.txt"
TFIDF_INDEX_PATH = "reuters/tf_idf/indice_invertido_tf_idf.txt"

# Leer las stopwords desde el archivo
with open(STOPWORDS_PATH, 'r', encoding='ascii') as file:
    stop_words = set(word.strip() for word in file.readlines())

# Cargar documentos preprocesados
documentos = {}
for filename in os.listdir(PROCESSED_PATH):
    filepath = os.path.join(PROCESSED_PATH, filename)
    with open(filepath, 'r', encoding='ascii') as file:
        cleaned_text = file.read()
        documentos[filename] = cleaned_text

# Vectorización Bag of Words
corpus = list(documentos.values())
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(corpus)
df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer_bow.get_feature_names_out(), index=documentos.keys())

# Vectorización TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(corpus)
df_tf_idf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out(), index=documentos.keys())

# Cargar índices invertidos desde archivos de texto
def load_inverted_index_from_txt(filepath):
    inverted_index = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        current_term = None
        for line in file:
            line = line.strip()
            if line.startswith("Termino:"):
                current_term = line.split("Termino: ")[1]
                inverted_index[current_term] = []
            elif line.startswith("Documento:"):
                doc_info = line.split("Documento: ")[1]
                doc_name, weight = doc_info.split(", Frecuencia: ")
                inverted_index[current_term].append((doc_name, float(weight)))
    return inverted_index

inverted_index_bow_loaded = load_inverted_index_from_txt(BOW_INDEX_PATH)
inverted_index_tfidf_loaded = load_inverted_index_from_txt(TFIDF_INDEX_PATH)

# Funciones de búsqueda
def process_query(query):
    cleaned_query = re.sub(r'[^\w\s]', '', query.lower())
    words = cleaned_query.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    filtered_words = [word for word in stemmed_words if word not in stop_words]
    return filtered_words

def jaccard_similarity(query_tokens, document_tokens):
    intersection = len(set(query_tokens) & set(document_tokens))
    union = len(set(query_tokens) | set(document_tokens))
    return intersection / union if union != 0 else 0

def cosine_similarity_score(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def search_with_bow(query, inverted_index_bow, documents):
    query_tokens = process_query(query)
    doc_tokens = {doc_id: documentos[doc_id].split() for doc_id in documentos}
    scores = {}
    for doc_id in doc_tokens:
        scores[doc_id] = jaccard_similarity(query_tokens, doc_tokens[doc_id])
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in ranked_results]

def search_with_tfidf(query, tfidf_matrix, vectorizer_tfidf, documents):
    query_tokens = process_query(query)
    query_vector = vectorizer_tfidf.transform([' '.join(query_tokens)]).toarray()[0]
    scores = {}
    for idx, doc_id in enumerate(documents.keys()):
        doc_vector = tfidf_matrix[idx].toarray()[0]
        scores[doc_id] = cosine_similarity_score(query_vector, doc_vector)
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in ranked_results]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    search_type = request.form['search_type']
    
    if search_type == 'bow':
        results_bow = search_with_bow(query, inverted_index_bow_loaded, documentos)
        return render_template('resultados.html', results_bow=results_bow[:5], results_tfidf=[])
    elif search_type == 'tfidf':
        results_tfidf = search_with_tfidf(query, X_tfidf, vectorizer_tfidf, documentos)
        return render_template('resultados.html', results_bow=[], results_tfidf=results_tfidf[:5])

if __name__ == '__main__':
    app.run(debug=True)
