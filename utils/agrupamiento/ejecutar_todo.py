"""
Script principal: orquesta la lectura, preprocesamiento, ejecución de los tres algoritmos jerárquicos
y la evaluación (correlación cophenética + silhouette). Finalmente guarda resultados y dendrogramas.

Salidas esperadas en data/processed/:
 - dendrograma_promedio.png
 - dendrograma_completo.png
 - dendrograma_ward.png
 - metricas_agrupamiento.csv

Ejecución:
    python -m utils.agrupamiento.ejecutar_todo
"""

from pathlib import Path
import logging
import csv
import math
import numpy as np
from typing import Optional
import sys

# Se añade la carpeta raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# --- Importaciones internas del proyecto ---
from utils.agrupamiento.reprocesar_y_vectorizar import (
    cargar_abstracts_desde_bib,
    preprocesar_abstracts,
    vectorizar_textos,
)

from utils.agrupamiento.algoritmos_jerarquicos import (
    ejecutar_average_linkage,
    ejecutar_complete_linkage,
    ejecutar_ward,
    evaluar_por_silhouette,
)

# --- Configuración del logger ---
logger = logging.getLogger("agrupamiento.ejecucion")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ------------------------------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------------------------------

def seleccionar_mejor_k(dic_silhouette: dict) -> Optional[int]:
    """
    Selecciona el valor de k con el mayor puntaje silhouette, ignorando NaN.
    """
    validos = {k: v for k, v in dic_silhouette.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if not validos:
        return None
    mejor_k = max(validos.keys(), key=lambda kk: validos[kk])
    return mejor_k


def convertir_a_denso(X):
    """
    Convierte una matriz dispersa (sparse) a densa con precaución.
    Muestra advertencia si el tamaño podría consumir mucha memoria.
    """
    try:
        from scipy.sparse import issparse
        if issparse(X):
            n, m = X.shape
            if n * m > 20_000_000:  # umbral heurístico
                logger.warning("Conversión a matriz densa puede ser costosa: n=%d, m=%d", n, m)
            return X.toarray()
    except Exception:
        pass
    return np.asarray(X)


# ------------------------------------------------------------------------------------
# Función principal
# ------------------------------------------------------------------------------------

def main_analizador_dendrogramas():
    # Definir rutas del proyecto
    raiz = Path(__file__).resolve().parents[2]
    ruta_bib = raiz / "static" / "data" / "processed" / "merged.bib"
    dir_salida = raiz / "static" / "salidas" / "agrupamiento_y_dendrogramas"
    dir_salida.mkdir(parents=True, exist_ok=True)

    # --- 1️⃣ Cargar y preprocesar abstracts ---
    abstracts_crudos = cargar_abstracts_desde_bib(ruta_bib)
    if not abstracts_crudos:
        logger.error("No se encontraron abstracts en %s. Abortando ejecución.", ruta_bib)
        return

    abstracts_limpios = preprocesar_abstracts(abstracts_crudos)
    claves, vectorizador, X = vectorizar_textos(abstracts_limpios)

    # --- 2️⃣ Ejecutar los tres algoritmos jerárquicos ---
    Z_avg, coph_avg = ejecutar_average_linkage(X, claves, dir_salida, p=10, max_label_len=60, orientacion="top")
    Z_comp, coph_comp = ejecutar_complete_linkage(X, claves, dir_salida, p=5, max_label_len=60, orientacion="top")
    Z_ward, coph_ward, svd = ejecutar_ward(X, claves, dir_salida, n_componentes=2, p=5, max_label_len=60, orientacion="top")

    # --- 3️⃣ Evaluar la calidad de agrupamientos con Silhouette ---
    rango_k = list(range(2, 9))
    sil_avg = evaluar_por_silhouette(X, Z_avg, "promedio", rango_k, metrica="cosine")
    sil_comp = evaluar_por_silhouette(X, Z_comp, "completo", rango_k, metrica="cosine")

    try:
        X_denso = convertir_a_denso(X)
        X_reducido = svd.transform(X_denso)
    except Exception:
        try:
            X_reducido = svd.transform(X)
        except Exception as ex:
            logger.warning("No se pudo obtener representación reducida para Ward: %s", ex)
            X_reducido = None

    sil_ward = evaluar_por_silhouette(X, Z_ward, "ward", rango_k, metrica="euclidean", X_reducido=X_reducido)

    # --- 4️⃣ Guardar métricas en CSV ---
    ruta_metricas = dir_salida / "metricas_agrupamiento.csv"
    with ruta_metricas.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["metodo", "correlacion_cophenetica", "k", "silhouette"])
        for k in rango_k:
            escritor.writerow(["promedio", coph_avg, k, sil_avg.get(k, "")])
        for k in rango_k:
            escritor.writerow(["completo", coph_comp, k, sil_comp.get(k, "")])
        for k in rango_k:
            escritor.writerow(["ward", coph_ward, k, sil_ward.get(k, "")])

    logger.info("Métricas guardadas en: %s", ruta_metricas)

    # --- 5️⃣ Mostrar resultados en logs ---
    def mostrar_mejor(nombre, dic_sil):
        mejor_k = seleccionar_mejor_k(dic_sil)
        if mejor_k is None:
            logger.info("No se encontró un valor válido de silhouette para %s.", nombre)
        else:
            logger.info("Mejor silhouette (%s): k=%s → score=%.4f", nombre, mejor_k, dic_sil.get(mejor_k, float('nan')))

    logger.info("Correlaciones cophenéticas → promedio: %.4f | completo: %.4f | ward: %.4f",
                coph_avg, coph_comp, coph_ward)

    mostrar_mejor("promedio", sil_avg)
    mostrar_mejor("completo", sil_comp)
    mostrar_mejor("ward", sil_ward)

    logger.info("Resultados finales almacenados en: %s", dir_salida)


# ------------------------------------------------------------------------------------
# Ejecución directa
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    main_analizador_dendrogramas()
