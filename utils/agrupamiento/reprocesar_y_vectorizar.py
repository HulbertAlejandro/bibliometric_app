"""
Preprocesamiento y vectorización de abstracts científicos.
- Lee el archivo 'merged.bib' generado por el proceso de unificación.
- Limpia, tokeniza, lematiza (usando NLTK) y elimina stopwords.
- Genera una representación vectorial mediante TF-IDF.
"""

from pathlib import Path
import re
import logging
from typing import List, Tuple

# Librerías para manejo de archivos bibtex, procesamiento de texto y vectorización
import bibtexparser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración de logging
logger = logging.getLogger("clustering.preprocesamiento")

# Intentamos descargar recursos de NLTK si no están disponibles localmente
try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Descargando recursos de NLTK requeridos (wordnet, stopwords, omw-1.4)...")
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("omw-1.4")

# Inicialización global (para evitar recarga)
LEMATIZADOR = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))


# ----------------------------------------------------------------------------------------------------
# Funciones principales
# ----------------------------------------------------------------------------------------------------

def cargar_abstracts_desde_bib(ruta_bib: Path) -> List[Tuple[str, str]]:
    """Lee el archivo 'merged.bib' y devuelve lista (ID, abstract)."""
    texto = ruta_bib.read_text(encoding="utf-8", errors="ignore")
    db = bibtexparser.loads(texto)
    resultados = []

    for entrada in db.entries:
        abstract = entrada.get("abstract", "").strip()
        clave = entrada.get("ID", entrada.get("key", ""))
        if abstract:
            resultados.append((clave, abstract))

    logger.info("Cargados %d abstracts desde %s", len(resultados), ruta_bib)
    return resultados


def limpieza_basica(texto: str) -> str:
    """Elimina URLs, caracteres especiales y normaliza espacios."""
    texto = re.sub(r"http\S+", " ", texto)
    texto = re.sub(r"[^A-Za-z0-9\s]", " ", texto)
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def lematizar_y_eliminar_stopwords(texto: str, lematizador: WordNetLemmatizer = LEMATIZADOR) -> str:
    """
    Tokeniza el texto, lematiza cada palabra y elimina stopwords.
    Retorna el texto limpio listo para vectorizar.
    """
    tokens = texto.split()
    lemas = []

    for t in tokens:
        if t in STOPWORDS:
            continue
        try:
            lema = lematizador.lemmatize(t)
        except Exception:
            lema = t
        if lema and len(lema) > 1:
            lemas.append(lema)

    return " ".join(lemas)


def preprocesar_abstracts(pares_crudos: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Aplica limpieza y lematización a cada abstract."""
    procesados = []
    for clave, texto in pares_crudos:
        limpio = limpieza_basica(texto)
        limpio = lematizar_y_eliminar_stopwords(limpio)
        if limpio:
            procesados.append((clave, limpio))
    logger.info("Preprocesados %d abstracts", len(procesados))
    return procesados


def vectorizar_textos(pares_limpios: List[Tuple[str, str]],
                      max_features: int = 20000,
                      ngram_range: tuple = (1, 2),
                      min_df: int = 2):
    """Vectoriza los textos preprocesados utilizando TF-IDF."""
    claves = [k for k, _ in pares_limpios]
    textos = [t for _, t in pares_limpios]

    vectorizador = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english"
    )

    X = vectorizador.fit_transform(textos)
    logger.info("TF-IDF generado. Dimensiones: %s", X.shape)
    return claves, vectorizador, X


# ----------------------------------------------------------------------------------------------------
# Ejemplo de uso
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ruta = Path("data/processed/merged.bib")
    if not ruta.exists():
        logger.error("No se encontró el archivo %s", ruta)
    else:
        abstracts = cargar_abstracts_desde_bib(ruta)
        procesados = preprocesar_abstracts(abstracts)
        claves, vectorizador, X = vectorizar_textos(procesados)
        logger.info("Vectorización completada con %d documentos.", len(claves))
