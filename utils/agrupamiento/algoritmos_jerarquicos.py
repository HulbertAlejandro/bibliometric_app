"""
Aplicación de los 3 algoritmos jerárquicos y cálculo de métricas de similitud.

Algoritmos implementados:
 - average linkage (distancia coseno)
 - complete linkage (distancia coseno)
 - ward (reducción SVD → distancia euclidiana → linkage ward)

Cada algoritmo:
 - Genera un dendrograma
 - Calcula la correlación cophenética
 - Permite calcular el coeficiente de silhouette

Las salidas gráficas y métricas se guardan en:
    salidas/agrupamiento_jerarquico/
"""

from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# Configuración del logger
logger = logging.getLogger("agrupamiento.algoritmos")

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------

def calcular_distancia_coseno(X):
    """
    Calcula la matriz de distancias coseno condensada.
    X: matriz TF-IDF (sparse o densa)
    Retorna: arreglo 1D con distancias (formato condensado de pdist)
    """
    try:
        from scipy.sparse import issparse
        if issparse(X):
            logger.debug("Convirtiendo matriz sparse a densa (esto puede usar mucha memoria).")
            Xd = X.toarray()
        else:
            Xd = np.asarray(X)
    except Exception:
        Xd = np.asarray(X)
    return pdist(Xd, metric="cosine")


def _acortar_etiquetas(labels: List[str], max_len: int = 60) -> List[str]:
    out = []
    for lab in labels:
        lab = str(lab)
        if len(lab) > max_len:
            out.append(lab[: max_len - 3].strip() + "...")
        else:
            out.append(lab)
    return out

def _ajustar_tamano_figura(n_leaves: int, orientacion: str = "top"):
    base = 6
    if orientacion == "top":
        width = min(40, max(8, n_leaves * 0.25))
        height = base
    else:
        width = base
        height = min(60, max(6, n_leaves * 0.25))
    return (width, height)

def _dibujar_dendrograma(
    Z,
    etiquetas: List[str],
    ruta_salida: Path,
    p: int = 40,
    max_label_len: int = 60,
    orientacion: str = "top",
    dpi: int = 200
) -> None:
    """
    Genera y guarda un dendrograma truncado (solo últimas ramas visibles).
    Llamada a dendrogram hecha con argumentos explícitos para evitar avisos de Pylance.
    """
    # Validaciones / coerciones simples
    if not isinstance(p, int):
        try:
            p = int(p)
        except Exception:
            p = 40

    orientacion = orientacion if orientacion in {"top", "bottom", "left", "right"} else "top"

    etiquetas_cortas = _acortar_etiquetas(etiquetas, max_len=max_label_len)
    tam_fig = _ajustar_tamano_figura(min(p, len(etiquetas_cortas)), orientacion)

    fig = plt.figure(figsize=tam_fig, dpi=dpi)

    # Llamada EXPLÍCITA a dendrogram (evita problemas de inferencia de tipos)
    try:
        if orientacion in ("right", "left"):
            # orientación vertical (hojas a la derecha/izquierda)
            dendrogram(
                Z,
                labels=etiquetas_cortas,
                truncate_mode="level",
                p=p,
                show_contracted=True,
                orientation=orientacion,
                leaf_rotation=0,
            )
        else:
            # orientación horizontal (hojas arriba/abajo)
            dendrogram(
                Z,
                labels=etiquetas_cortas,
                truncate_mode="level",
                p=p,
                show_contracted=True,
                orientation=orientacion,
                leaf_rotation=90,
            )

        plt.tight_layout()
        fig.savefig(ruta_salida, dpi=dpi)
        logger.info("Dendrograma guardado en: %s", ruta_salida)
    except Exception as ex:
        logger.exception("Error al dibujar dendrograma: %s", ex)
    finally:
        plt.close(fig)

# -------------------------------------------------------------------
# Algoritmos de agrupamiento jerárquico
# -------------------------------------------------------------------

def ejecutar_average_linkage(X, etiquetas: List[str], dir_salida: Path, p: int = 40):
    """
    Ejecuta el algoritmo Average Linkage usando distancia coseno.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    D = calcular_distancia_coseno(X)
    Z = linkage(D, method="average")
    corr, _ = cophenet(Z, D)
    logger.info("Average linkage: correlación cophenética = %.4f", corr)

    try:
        _dibujar_dendrograma(Z, etiquetas, dir_salida / "dendrograma_average.png", p=p)
    except Exception as ex:
        logger.warning("Error al generar dendrograma (average): %s", ex)
    return Z, corr


def ejecutar_complete_linkage(X, etiquetas: List[str], dir_salida: Path, p: int = 40):
    """
    Ejecuta el algoritmo Complete Linkage usando distancia coseno.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    D = calcular_distancia_coseno(X)
    Z = linkage(D, method="complete")
    corr, _ = cophenet(Z, D)
    logger.info("Complete linkage: correlación cophenética = %.4f", corr)

    try:
        _dibujar_dendrograma(Z, etiquetas, dir_salida / "dendrograma_complete.png", p=p)
    except Exception as ex:
        logger.warning("Error al generar dendrograma (complete): %s", ex)
    return Z, corr


def ejecutar_ward(X, etiquetas: List[str], dir_salida: Path, n_componentes: int = 50, p: int = 40):
    """
    Ejecuta el algoritmo Ward, aplicando reducción SVD previa.
    Utiliza distancia euclidiana.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    X_reducido = svd.fit_transform(X)
    Z = linkage(X_reducido, method="ward")
    D_euc = pdist(X_reducido, metric="euclidean")
    corr, _ = cophenet(Z, D_euc)
    logger.info("Ward linkage: correlación cophenética = %.4f (SVD componentes=%d)", corr, n_componentes)

    try:
        _dibujar_dendrograma(Z, etiquetas, dir_salida / "dendrograma_ward.png", p=p)
    except Exception as ex:
        logger.warning("Error al generar dendrograma (ward): %s", ex)
    return Z, corr, svd

# -------------------------------------------------------------------
# Evaluación mediante Silhouette
# -------------------------------------------------------------------

def evaluar_por_silhouette(X, Z, metodo: str, rango_k: List[int],
                           metrica: str = "cosine", X_reducido: Optional[np.ndarray] = None) -> Dict[int, float]:
    """
    Evalúa la calidad de los clusters para varios valores de k usando el coeficiente de silhouette.
    Retorna un diccionario {k: valor_silhouette}.
    """
    resultados: Dict[int, float] = {}
    omitidos: List[int] = []

    for k in rango_k:
        if k < 2:
            resultados[k] = np.nan
            continue
        try:
            etiquetas = fcluster(Z, t=k, criterion="maxclust")
            if len(set(etiquetas)) < 2:
                resultados[k] = np.nan
                continue

            if X_reducido is not None:
                score = silhouette_score(X_reducido, etiquetas, metric="euclidean")
            else:
                Xd = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
                score = silhouette_score(Xd, etiquetas, metric=metrica)
            resultados[k] = float(score)
        except Exception as ex:
            resultados[k] = np.nan
            omitidos.append(k)
            logger.debug("Silhouette error método=%s, k=%d: %s", metodo, k, ex)

    if omitidos:
        logger.info("Valores de k omitidos para %s (inválidos): %s", metodo, omitidos)

    return resultados
