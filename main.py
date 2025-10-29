import subprocess
import time
import subprocess
from scraper.acm_scraper import extraer_acm_bibtex
from scraper.ieee_scraper import extraer_ieee_bibtex_chrome
from utils.analizador_palabras_clave import main_analizador_palabras_clave
from utils.agrupamiento.ejecutar_todo import main_analizador_dendrogramas

def run_acm_scraper():
    try:
        start = int(input("Página de inicio (ej: 0): "))
        count = int(input("¿Cuántas páginas desea scrapear?: "))
        for i in range(start, start + count):
            print(f"\n>>> Scraping page {i}")
            extraer_acm_bibtex(i)
            time.sleep(3)  # evita sobrecargar el servidor
    except ValueError:
        print("Entrada inválida, por favor ingrese números.")

def run_ieee_scraper():
    try:
        start = int(input("Página de Inicio (e.g. 1): "))
        count = int(input("¿Cuántas páginas desea scrapear?: "))

        for i in range(start, start + count):
            print(f"\n>>> Scraping página {i}")
            extraer_ieee_bibtex_chrome(i)
            time.sleep(3)  # evita sobrecargar el servidor

    except ValueError:
        print("Entrada inválida, por favor ingrese números.")

def run_merge_bib():
    print("\n=== MERGE BIB ===")
    ieee_dir = "data/raw/IEEE"
    acm_dirs = "data/raw/ACM"
    out_dir = "data/processed"

    subprocess.run([
        "python", "utils/merge_bib.py",
        "--ieee-dir", ieee_dir,
        "--acm-dirs", acm_dirs,
        "--out-dir", out_dir
    ])

def run_compare_articles():
    ids = input("Ingrese los IDs separados por coma (ej: merged1,merged3): ")
    subprocess.run(["python", "utils/comparar_articulos.py", "--compare_ids", ids])

def run_keyword_analyzer():
    main_analizador_palabras_clave()
    print("\n Análisis completado. Gráficas generadas en la carpeta de salidas.")

def run_dendrograms_analyzer():
    main_analizador_dendrogramas()
    print("\n Análisis completado. Dendrogramas generados en la carpeta de salidas.")

def menu():
    while True:
        print("\n===== MENÚ PRINCIPAL =====")
        print("1. Scraper ACM")
        print("2. Scraper IEEE")
        print("3. Merge Bib")
        print("4. Comparar artículos")
        print("5. Analizador de Keywords")
        print("6. Analizador de Similitud con Dendrogramas")
        print("7. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            run_acm_scraper()
        elif opcion == "2":
            run_ieee_scraper()
        elif opcion == "3":
            run_merge_bib()
        elif opcion == "4":
            run_compare_articles()
        elif opcion == "5":
            run_keyword_analyzer()
        elif opcion == "6":
            run_dendrograms_analyzer()
        elif opcion == "7":
            print("Terminando...")
            break
        else:
            print("Opción no válida, intenta de nuevo.")

if __name__ == "__main__":
    menu()