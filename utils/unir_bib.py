"""
Script para fusionar entradas bibliográficas de IEEE y ACM con detección de duplicados.
- Orden de emparejamiento: ISBN -> título difuso (fuzzy)
- Salidas:
    * merged.bib: archivo con entradas únicas fusionadas
    * duplicates.bib: archivo con entradas duplicadas que fueron eliminadas
    * merge_map.csv: reporte con el mapeo (tipo de match, score, archivos fuente)
Uso:
    python utils/unir_bib.py --ieee-dir data/raw/IEEE --acm-dirs data/raw/ACM --out-dir data/processed
"""

from __future__ import annotations
import argparse
import csv
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
from rapidfuzz import process, fuzz

# ---- Configuración de logging ----
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("merge_bib")


# ---- Utilidades ----
def raiz_proyecto() -> Path:
    """Retorna la raíz del proyecto."""
    return Path(__file__).resolve().parent.parent


def leer_archivos_bib(folder: Path) -> List[Dict]:
    """
    Lee todos los archivos .bib en una carpeta y retorna una lista de entradas.
    Agrega el nombre del archivo fuente a cada entrada.
    """
    entries: List[Dict] = []
    if not folder.exists():
        logger.warning("Carpeta no existe: %s", folder)
        return entries
    
    for p in sorted(folder.glob("*.bib")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            db = bibtexparser.loads(text)
            for e in db.entries:
                e.setdefault("_source_file", str(p.name))
                e.setdefault("_source_folder", str(folder.name))
                entries.append(e)
        except Exception as ex:
            logger.error("Error leyendo %s: %s", p, ex)
    
    logger.info("Leídos %d entries desde %s", len(entries), folder)
    return entries


def normalizar_isbn(isbn_raw: Optional[str]) -> Optional[List[str]]:
    """Normaliza el campo ISBN/ISSN."""
    if not isbn_raw:
        return None
    s = re.sub(r'[{}\s"]', "", isbn_raw)
    parts = re.split(r'[;,/|]', s)
    parts = [p.strip().lower() for p in parts if p.strip()]
    return parts if parts else None


def normalizar_titulo(title: Optional[str]) -> str:
    """Normaliza el título para comparación difusa."""
    if not title:
        return ""
    t = title.lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extraer_isbns_de_entrada(e: Dict) -> Optional[List[str]]:
    """Extrae y normaliza el ISBN/ISSN de una entrada."""
    for key in ("isbn", "ISBN", "issn", "ISSN"):
        if key in e and str(e[key]).strip():
            return normalizar_isbn(str(e[key]))
    return None


def unir_entradas(e_list: List[Dict]) -> Dict:
    """
    Fusiona una lista de entradas bib en una sola.
    Estrategia:
      - ENTRYTYPE del primer elemento
      - keywords: une tokens únicos separados por punto y coma (;)
      - abstract: el más largo
      - otros campos: el valor más largo no vacío
    """
    merged: Dict = {}
    merged["ENTRYTYPE"] = e_list[0].get("ENTRYTYPE", e_list[0].get("type", "inproceedings"))
    
    # Guardar información de fuentes originales
    source_files = []
    source_ids = []
    for e in e_list:
        if "_source_file" in e:
            source_files.append(e["_source_file"])
        if "ID" in e:
            source_ids.append(e["ID"])
    
    merged["_original_sources"] = "; ".join(source_files)
    merged["_original_ids"] = "; ".join(source_ids)
    merged["_merge_count"] = str(len(e_list))
    
    fields = set().union(*[set(e.keys()) for e in e_list])
    fields = [f for f in fields if not f.startswith("_")]
    
    for f in fields:
        vals = [str(e.get(f, "")).strip() for e in e_list if str(e.get(f, "")).strip()]
        if not vals:
            continue
        
        if f.lower() == "keywords":
            kwset = set()
            for v in vals:
                for tok in re.split(r"[;,]", v):
                    tok = tok.strip()
                    if tok:
                        kwset.add(tok)
            merged[f] = "; ".join(sorted(kwset))
        elif f.lower() == "abstract":
            merged[f] = max(vals, key=len)
        elif f.lower() == "author":
            authors = set()
            for v in vals:
                for author in re.split(r"\s+and\s+", v, flags=re.IGNORECASE):
                    author = author.strip()
                    if author:
                        authors.add(author)
            merged[f] = " and ".join(sorted(authors))
        else:
            merged[f] = max(vals, key=len)
    
    return merged

# ---- Lógica principal de merge ----
def construir_mapa_isbn(entries: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
    """Construye un mapa ISBN -> lista de (índice, entrada)."""
    m: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, e in enumerate(entries):
        isbns = extraer_isbns_de_entrada(e)
        if isbns:
            for isb in isbns:
                m.setdefault(isb, []).append((idx, e))
    return m


def unir_colecciones(
    ieee_entries: List[Dict],
    acm_entries: List[Dict],
    title_threshold: int = 88,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Fusiona colecciones IEEE y ACM detectando duplicados.
    Retorna:
        - Lista de entradas fusionadas únicas
        - Lista de entradas duplicadas que fueron eliminadas
        - Lista de mapeo para reporte CSV
    """
    used_acm = set()
    used_ieee = set()
    merged_results: List[Dict] = []
    duplicate_entries: List[Dict] = []
    mapping_rows: List[Dict] = []

    acm_isbn_map = construir_mapa_isbn(acm_entries)
    
    # 1) Emparejamiento por ISBN
    for i_idx, e_ieee in enumerate(ieee_entries):
        isbns = extraer_isbns_de_entrada(e_ieee)
        matched = False
        if isbns:
            for isb in isbns:
                if isb in acm_isbn_map:
                    acm_hits = acm_isbn_map[isb]
                    for a_idx, a_entry in acm_hits:
                        if a_idx in used_acm:
                            continue
                        
                        # Fusionar entradas
                        merged = unir_entradas([e_ieee, a_entry])
                        merged_results.append(merged)
                        
                        # Guardar duplicado (la entrada ACM que se fusionó)
                        dup_entry = a_entry.copy()
                        dup_entry["_duplicate_reason"] = f"Duplicado con IEEE por ISBN: {isb}"
                        dup_entry["_merged_with"] = e_ieee.get("ID", "")
                        duplicate_entries.append(dup_entry)
                        
                        used_acm.add(a_idx)
                        used_ieee.add(i_idx)
                        
                        mapping_rows.append({
                            "merged_key": "",
                            "ieee_key": e_ieee.get("ID", ""),
                            "acm_key": a_entry.get("ID", ""),
                            "match_type": "ISBN",
                            "score": 100,
                            "isbn": isb,
                            "ieee_file": e_ieee.get("_source_file", ""),
                            "acm_file": a_entry.get("_source_file", ""),
                        })
                        matched = True
                    if matched:
                        break

    # Preparar pools para comparación difusa
    remaining_ieee = [(i, e) for i, e in enumerate(ieee_entries) if i not in used_ieee]
    remaining_acm = [(i, e) for i, e in enumerate(acm_entries) if i not in used_acm]

    acm_pool_idx: List[int] = []
    acm_pool_titles: List[str] = []
    for idx, e in remaining_acm:
        acm_pool_idx.append(idx)
        acm_pool_titles.append(normalizar_titulo(e.get("title", "")))

    # 2) Emparejamiento difuso por título
    for i_idx, e_ieee in remaining_ieee:
        title_ieee = normalizar_titulo(e_ieee.get("title", ""))
        if not title_ieee or not acm_pool_titles:
            continue
        
        best = process.extractOne(title_ieee, acm_pool_titles, scorer=fuzz.token_set_ratio)
        if best:
            candidate_title, score, pos = best
            if score >= title_threshold:
                acm_real_idx = acm_pool_idx[pos]
                e_acm = acm_entries[acm_real_idx]
                
                # Fusionar entradas
                merged = unir_entradas([e_ieee, e_acm])
                merged_results.append(merged)
                
                # Guardar duplicado
                dup_entry = e_acm.copy()
                dup_entry["_duplicate_reason"] = f"Duplicado con IEEE por título (score: {score})"
                dup_entry["_merged_with"] = e_ieee.get("ID", "")
                duplicate_entries.append(dup_entry)
                
                used_acm.add(acm_real_idx)
                used_ieee.add(i_idx)
                
                mapping_rows.append({
                    "merged_key": "",
                    "ieee_key": e_ieee.get("ID", ""),
                    "acm_key": e_acm.get("ID", ""),
                    "match_type": "TITLE",
                    "score": int(score),
                    "isbn": ";".join(extraer_isbns_de_entrada(e_ieee) or []) or ";".join(extraer_isbns_de_entrada(e_acm) or []),
                    "ieee_file": e_ieee.get("_source_file", ""),
                    "acm_file": e_acm.get("_source_file", ""),
                })
                
                # Eliminar del pool
                del acm_pool_titles[pos]
                del acm_pool_idx[pos]

    # 3) Agregar entradas no emparejadas
    for i_idx, e_ieee in enumerate(ieee_entries):
        if i_idx not in used_ieee:
            merged_results.append(unir_entradas([e_ieee]))
            mapping_rows.append({
                "merged_key": "",
                "ieee_key": e_ieee.get("ID", ""),
                "acm_key": "",
                "match_type": "UNMATCHED_IEEE",
                "score": 0,
                "isbn": ";".join(extraer_isbns_de_entrada(e_ieee) or []),
                "ieee_file": e_ieee.get("_source_file", ""),
                "acm_file": "",
            })
    
    for a_idx, e_acm in enumerate(acm_entries):
        if a_idx not in used_acm:
            merged_results.append(unir_entradas([e_acm]))
            mapping_rows.append({
                "merged_key": "",
                "ieee_key": "",
                "acm_key": e_acm.get("ID", ""),
                "match_type": "UNMATCHED_ACM",
                "score": 0,
                "isbn": ";".join(extraer_isbns_de_entrada(e_acm) or []),
                "ieee_file": "",
                "acm_file": e_acm.get("_source_file", ""),
            })

    return merged_results, duplicate_entries, mapping_rows


# ---- Escritura de archivos ----
def escribir_archivos(
    out_dir: Path, 
    merged_results: List[Dict], 
    duplicate_entries: List[Dict],
    mapping_rows: List[Dict], 
    out_bib: str, 
    out_dup_bib: str,
    out_csv: str
) -> None:
    """
    Escribe tres archivos:
    1. merged.bib - Entradas únicas fusionadas
    2. duplicates.bib - Entradas duplicadas eliminadas
    3. merge_map.csv - Reporte de mapeo
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. Archivo de entradas únicas =====
    out_entries = []
    for idx, e in enumerate(merged_results, start=1):
        key_base = f"merged{idx}"
        e["ID"] = key_base
        out_entries.append(e)
    
    # Actualizar merged_key en el mapeo
    for i, row in enumerate(mapping_rows):
        if i < len(out_entries):
            row["merged_key"] = out_entries[i]["ID"]

    bibdb = BibDatabase()
    bibdb.entries = out_entries
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = ('ID',)
    
    out_bib_path = out_dir / out_bib
    out_bib_path.write_text(writer.write(bibdb), encoding="utf-8")
    logger.info("Archivo unificado guardado: %s (%d entradas)", out_bib_path, len(out_entries))
    
    # ===== 2. Archivo de duplicados =====
    dup_entries = []
    for idx, e in enumerate(duplicate_entries, start=1):
        key_base = f"duplicate{idx}"
        e["ID"] = key_base
        dup_entries.append(e)
    
    dup_bibdb = BibDatabase()
    dup_bibdb.entries = dup_entries
    
    out_dup_path = out_dir / out_dup_bib
    out_dup_path.write_text(writer.write(dup_bibdb), encoding="utf-8")
    logger.info("Archivo de duplicados guardado: %s (%d entradas)", out_dup_path, len(dup_entries))
    
    # ===== 3. CSV de mapeo =====
    out_csv_path = out_dir / out_csv
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["merged_key", "ieee_key", "acm_key", "match_type", "score", "isbn", "ieee_file", "acm_file"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in mapping_rows:
            w.writerow(r)
    logger.info("CSV de mapeo guardado: %s", out_csv_path)
    
    # ===== Resumen =====
    logger.info("="*60)
    logger.info("RESUMEN DE UNIFICACIÓN")
    logger.info("="*60)
    logger.info("Total entradas únicas: %d", len(out_entries))
    logger.info("Total duplicados eliminados: %d", len(dup_entries))
    logger.info("Total entradas procesadas: %d", len(out_entries) + len(dup_entries))
    
    # Estadísticas de match
    match_types = {}
    for row in mapping_rows:
        mt = row["match_type"]
        match_types[mt] = match_types.get(mt, 0) + 1
    
    logger.info("\nEstadísticas de emparejamiento:")
    for mt, count in sorted(match_types.items()):
        logger.info("  - %s: %d", mt, count)
    logger.info("="*60)


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    pr = raiz_proyecto()
    parser = argparse.ArgumentParser(
        description="Merge IEEE and ACM .bib files, eliminando duplicados y generando archivo de duplicados"
    )
    parser.add_argument("--ieee-dir", default=str(pr / "static" / "data" / "raw" / "IEEE"), help="Carpeta con archivos .bib IEEE")
    parser.add_argument(
        "--acm-dirs",
        default="static/data/raw/ACM,static/data/raw/ACM2,static/data/raw/ACM3",
        help="Carpeta(s) con archivos .bib ACM, separadas por comas",
    )
    parser.add_argument("--out-dir", default=str(pr / "static" / "data" / "processed"), help="Carpeta de salida")
    parser.add_argument("--out-bib", default="merged.bib", help="Nombre del .bib unificado")
    parser.add_argument("--out-dup-bib", default="duplicates.bib", help="Nombre del .bib de duplicados")
    parser.add_argument("--out-csv", default="merge_map.csv", help="CSV con el mapa de merges")
    parser.add_argument("--title-threshold", type=int, default=88, help="Umbral (0-100) para fuzzy title match")
    return parser.parse_args()


def main():
    args = parse_args()
    pr = raiz_proyecto()
    
    # Resolver rutas
    ieee_dir = Path(args.ieee_dir)
    if not ieee_dir.is_absolute():
        ieee_dir = (pr / args.ieee_dir).resolve()
    
    acm_dirs_raw = args.acm_dirs.split(",")
    acm_dirs: List[Path] = []
    for ad in acm_dirs_raw:
        p = Path(ad.strip())
        if not p.is_absolute():
            p = (pr / ad.strip()).resolve()
        acm_dirs.append(p)
    
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (pr / args.out_dir).resolve()
    
    # Leer archivos
    ieee_entries = leer_archivos_bib(ieee_dir)
    acm_entries_all: List[Dict] = []
    for acm_dir in acm_dirs:
        acm_entries_all.extend(leer_archivos_bib(acm_dir))
    
    # Fusionar y detectar duplicados
    logger.info("\nProcesando unificación y detección de duplicados...")
    merged_results, duplicate_entries, mapping_rows = unir_colecciones(
        ieee_entries, 
        acm_entries_all, 
        title_threshold=args.title_threshold
    )
    
    # Escribir archivos
    logger.info("\nEscribiendo archivos...")
    escribir_archivos(
        out_dir, 
        merged_results, 
        duplicate_entries,
        mapping_rows, 
        args.out_bib, 
        args.out_dup_bib,
        args.out_csv
    )


if __name__ == "__main__":
    main()
