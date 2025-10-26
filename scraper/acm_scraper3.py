# acm_scraper3.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from pathlib import Path
import shutil
import time
import os


def extraer_acm_por_pagina(pagina_inicio: int, carpeta_descargas, ruta_perfil):
    """
    Abre la Biblioteca Digital de ACM en la página indicada, selecciona todos los resultados
    y descarga las citas en formato BibTeX. El navegador se abre y se cierra por cada página.
    """
    carpeta_salida = Path("data/raw/ACM3").resolve()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    Path(carpeta_descargas).mkdir(parents=True, exist_ok=True)

    opciones = configurar_navegador(carpeta_descargas, ruta_perfil)

    print(f"\nIniciando navegador para la página {pagina_inicio}...")
    navegador = uc.Chrome(options=opciones)
    espera = WebDriverWait(navegador, 20)

    # Construir la URL con la página correspondiente
    url = f"https://dl.acm.org/action/doSearch?AllField=generative+artificial+intelligence&startPage={pagina_inicio}&pageSize=50"
    navegador.get(url)

    try:
        print("Esperando para aceptar las cookies...")
        try:
            boton_cookies = espera.until(
                EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))
            )
            boton_cookies.click()
            print("Cookies aceptadas correctamente.")
        except:
            print("No se encontró el botón de cookies (posiblemente ya fueron aceptadas).")

        print("Seleccionando todos los resultados de la página...")
        casilla_todo = navegador.find_element(By.CSS_SELECTOR, "input[name='markall']")
        navegador.execute_script("arguments[0].click();", casilla_todo)
        time.sleep(2)

        print("Abriendo la ventana de exportación de citas...")
        boton_exportar = espera.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.export-citation")))
        navegador.execute_script("arguments[0].click();", boton_exportar)
        time.sleep(5)

        print("Descargando archivo BibTeX...")
        boton_descarga = espera.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.download__btn[title='Download citation']"))
        )
        navegador.execute_script("arguments[0].click();", boton_descarga)
        print("✅ Descarga iniciada correctamente.")

        # Esperar a que se complete la descarga
        time.sleep(10)
        archivos_bib = sorted(Path(carpeta_descargas).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not archivos_bib:
            print("❌ No se encontró ningún archivo .bib en la carpeta de descargas.")
        else:
            archivo_reciente = archivos_bib[0]
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_final = carpeta_salida / f"acm_pagina_{pagina_inicio}_{marca_tiempo}.bib"
            shutil.move(str(archivo_reciente), ruta_final)
            print(f"✅ Archivo BibTeX guardado en: {ruta_final}")

    except Exception as error:
        print(f"⚠️ Error durante la extracción de la página {pagina_inicio}: {error}")

    finally:
        print(f"Cerrando navegador para la página {pagina_inicio}...")
        navegador.quit()
        time.sleep(3)  # Evita sobrecargar el servidor


def configurar_navegador(carpeta_descargas, ruta_perfil):
    """
    Configura Google Chrome en modo sigiloso utilizando un perfil guardado
    (para evitar CAPTCHAs y mantener sesión activa).
    """
    opciones = uc.ChromeOptions()
    preferencias = {
        "download.default_directory": carpeta_descargas,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    opciones.add_experimental_option("prefs", preferencias)
    opciones.add_argument(rf"--user-data-dir={ruta_perfil}")
    opciones.add_argument(r"--profile-directory=Default")
    opciones.add_argument("--start-maximized")
    opciones.add_argument("--disable-popup-blocking")
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-blink-features=AutomationControlled")
    return opciones


if __name__ == "__main__":
    try:
        inicio = int(input("Número de página inicial (por ejemplo, 0): "))
        cantidad = int(input("¿Cuántas páginas deseas extraer?: "))

        carpeta_descargas = str(Path("downloads").resolve())
        ruta_perfil = r"C:\Users\hulbe\ChromeProfiles\scraper_profile"

        for i in range(inicio, inicio + cantidad):
            extraer_acm_por_pagina(i, carpeta_descargas, ruta_perfil)

        print("\n✅ Extracción finalizada correctamente.")
    except ValueError:
        print("❌ Entrada no válida. Asegúrate de escribir números enteros.")
