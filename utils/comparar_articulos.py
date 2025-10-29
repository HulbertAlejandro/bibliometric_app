import bibtexparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textdistance
from sentence_transformers import SentenceTransformer
import argparse

""""
Ejemplo de entrada:
python utils/comparar_articulos.py --compare_ids merged1,merged2
"""
def cargar_bib(path="static/data/processed/merged.bib"):
    with open(path, encoding="utf-8") as f:
        db = bibtexparser.load(f)
        return {entry["ID"]: entry.get("abstract", "") for entry in db.entries}
    
def levenshtein_sim(s1, s2):
    return 1 - textdistance.levenshtein.normalized_distance(s1, s2)

def jaccard_sim(s1, s2):
    set1, set2 = set(s1.lower().split()), set(s2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def dice_sim(s1, s2):
    set1, set2 = set(s1.lower().split()), set(s2.lower().split())
    return 2 * len(set1 & set2) / (len(set1) + len(set2))

def tfidf_cosine(s1, s2): 
    vec = TfidfVectorizer().fit([s1, s2])
    tfidf = vec.transform([s1, s2])
    return float(cosine_similarity(tfidf)[0][1])

def sbert_cosine(s1, s2):
    """"
    - Modelo más pequeño y rápido, con menos parámetros.
    - Ideal si priorizas velocidad sobre precisión.
    - Recomendado para aplicaciones en tiempo real o con muchos textos.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2") # Carga un modelo SBERT ligero
    # Genera (vectores) para los abstracts s1 y s2
    emb = model.encode([s1, s2], convert_to_numpy=True)
    # Calcula la similitud coseno
    return float(cosine_similarity(emb)[0][1])

def comparar_articulos_sbert(s1, s2):
    """"
    - Modelo más grande y preciso, entrenado para similitud semántica.
    - Captura relaciones semánticas profundas.
    - Segun lo investigado se utiliza para tareas de semantic textual similarity (STS).
    """
    # Usamos el mismo modelo potente
    sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Generamos (vectores) de los dos textos
    embeddings = sbert_model.encode([s1, s2], convert_to_numpy=True)
    # Calculamos la similitud coseno entre los abstracts
    sim = float(cosine_similarity(embeddings)[0][1])
    return sim

""""
Función para comparar (todos con todos) los articulos seleccionados para su comparación
La comparación se hace con todas la funciones requeridas
    :param abstracts: diccionario con abstracts, ej. {"merged1": "...", "merged2": "..."}
    :param article_ids: lista de IDs a comparar, ej. ["merged1", "merged2"]
"""
def comparar_articulos(abstracts, article_ids):
    n = len(article_ids)
    output = []
    for i in range(n):
        for j in range(i+1, n):
            s1, s2 = abstracts[article_ids[i]], abstracts[article_ids[j]]
            section = [f""]
            section.append(f" Levenshtein: {levenshtein_sim(s1, s2)}")
            section.append(f" Jaccard: {jaccard_sim(s1, s2)}")
            section.append(f" Dice: {dice_sim(s1, s2)}")
            section.append(f" TF-IDF Cosine: {tfidf_cosine(s1, s2)}")
            section.append(f" SBERT Cosine 1: {sbert_cosine(s1, s2)}")
            section.append(f" SBERT Cosine 2: {comparar_articulos_sbert(s1, s2)}")
            output.append('\n'.join(section))
    return '\n\n'.join(output) if output else "Sin comparaciones calculadas."
            
# main
if __name__ == "__main__":
    # Configuración de los parametros para indicar los articulos a comparar
    parser = argparse.ArgumentParser(description="Herramienta de comparación de artículos")
    parser.add_argument( "--compare_ids",
                        type=str,
                        help="IDs de artículos a comparar, separados por coma. Ej: --compare_ids merged1,merged2" )
    
    # Se instancian los parametros
    args = parser.parse_args()
    # Se cargan los abstracts del 'unir_bib.py'
    abstracts = cargar_bib()
    # Se obtienen los ids de los articulos que se quieren comparar
    if args.compare_ids:
        ids = args.compare_ids.split(",")
        comparar_articulos(abstracts, ids)