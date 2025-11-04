#acm_scraper.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from pathlib import Path
import shutil
import time
import os

def extraer_acm_bibtex(navegador, espera, carpeta_descargas, pagina_inicio: int, primera_pagina, ultima_pagina):
    """
    Extrae artículos de la biblioteca digital de ACM y descarga las citas en formato BibTeX.
    """
    carpeta_salida = Path("static/data/raw/ACM")
    Path(carpeta_descargas).mkdir(parents=True, exist_ok=True)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Ir a la página de resultados de ACM
    if primera_pagina:
        url = f"https://dl.acm.org/action/doSearch?AllField=generative+artificial+intelligence&startPage={pagina_inicio}&pageSize=100&sortBy=relevance"
        navegador.get(url)

    try:
        # Aceptar cookies si es la primera página
        if primera_pagina:
            print("Esperando para aceptar las cookies...")
            try:
                boton_cookies = espera.until(
                    EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))
                )
                boton_cookies.click()
                print("Cookies aceptadas correctamente.")
            except Exception:
                print("No se encontró el botón de cookies (posiblemente ya aceptadas o no visibles).")

        print("Seleccionando todos los resultados de la página...")
        casilla_todo = navegador.find_element(By.CSS_SELECTOR, "input[name='markall']")
        navegador.execute_script("arguments[0].click();", casilla_todo)
        time.sleep(2)

        print("Abriendo la ventana de exportación de citas...")
        boton_exportar = espera.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.export-citation"))
        )
        navegador.execute_script("arguments[0].click();", boton_exportar)
        time.sleep(5)
        if not primera_pagina:
            time.sleep(10)

        print("Descargando archivo BibTeX...")
        boton_descarga = espera.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.download__btn[title='Download citation']"))
        )
        navegador.execute_script("arguments[0].click();", boton_descarga)
        print("Descarga completada.")

        time.sleep(5)

        # Mover el archivo .bib descargado a la carpeta de salida
        archivos_bib = sorted(Path(carpeta_descargas).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not archivos_bib:
            print("No se encontró ningún archivo .bib en la carpeta de descargas.")
            return

        archivo_reciente = archivos_bib[0]
        marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_final = carpeta_salida / f"acm_descargado_{marca_tiempo}.bib"
        shutil.move(str(archivo_reciente), ruta_final)

        print(f"Archivo BibTeX guardado en: {ruta_final}")

        # Cerrar el cuadro de exportación
        try:
            boton_cerrar = espera.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "i.icon-close_thin"))
            )  
            navegador.execute_script("arguments[0].click();", boton_cerrar)
            print("Ventana de exportación cerrada.")
        except Exception:
            print("No se encontró el botón para cerrar el cuadro de exportación (posiblemente ya cerrado).")

        time.sleep(1)

        # Si no es la última página, pasar a la siguiente
        if not ultima_pagina:
            print("Pasando a la siguiente página de resultados...")
            boton_siguiente = espera.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pagination__btn--next"))
            )
            navegador.execute_script("arguments[0].click();", boton_siguiente)
            time.sleep(3)

    except Exception as error:
        print(f"Error durante la extracción de la página {pagina_inicio}: {error}")


def configurar_navegador(carpeta_descargas, ruta_perfil):
    """
    Configura Chrome en modo indetectable usando un perfil de usuario existente.
    """
    opciones = uc.ChromeOptions()
    preferencias = {
        "download.default_directory": carpeta_descargas,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    opciones.add_experimental_option("prefs", preferencias)

    # Reutilizar tu perfil real (donde ya pasaste el captcha)
    opciones.add_argument(rf"--user-data-dir={ruta_perfil}")  
    opciones.add_argument(r"--profile-directory=Default")     # o "Profile 1" según el perfil
    opciones.add_argument("--start-maximized")
    opciones.add_argument("--disable-popup-blocking")
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-blink-features=AutomationControlled")

    return opciones


if __name__ == "__main__":
    try:
        inicio = int(input("Número de página inicial (por ejemplo, 0): "))
        cantidad = int(input("¿Cuántas páginas deseas extraer?: "))

        # Configurar rutas
        carpeta_descargas = str(Path("downloads").resolve())

        # Ajusta esta ruta según tu perfil de Chrome
        ruta_perfil = str(Path.home() / "ChromeProfiles" / "scraper_profile")

        opciones = configurar_navegador(carpeta_descargas, ruta_perfil)

        print("Iniciando navegador en modo sigiloso con perfil guardado...")
        navegador = uc.Chrome(options=opciones, version_main=141)
        espera = WebDriverWait(navegador, 20)

        # Recorrer las páginas
        for i in range(inicio, inicio + cantidad):
            print(f"\n>>> Extrayendo información de la página {i}")
            extraer_acm_bibtex(
                navegador, espera, carpeta_descargas,
                pagina_inicio=i,
                primera_pagina=(i == inicio),
                ultima_pagina=(i == inicio + cantidad - 1)
            )
            time.sleep(3)

        input("Presiona Enter para cerrar el navegador...")
        navegador.quit()

    except ValueError:
        print("Entrada no válida. Asegúrate de escribir números enteros.")
