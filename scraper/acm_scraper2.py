# acm_scraper2.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from pathlib import Path
import shutil
import time
import os


def extraer_todo_acm_bibtex(navegador, espera, carpeta_descargas):
    """
    Abre la Biblioteca Digital de ACM, realiza una búsqueda predefinida y descarga
    todas las citas en formato BibTeX haciendo clic en 'Download now!' cuando aparece.
    """
    carpeta_salida = Path("data/raw/ACM2").resolve()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    Path(carpeta_descargas).mkdir(parents=True, exist_ok=True)

    palabra_clave = "generative artificial intelligence"
    print(f"Iniciando Chrome con perfil guardado... (buscando: {palabra_clave})")

    url = f"https://dl.acm.org/action/doSearch?AllField={palabra_clave.replace(' ', '+')}&startPage=0&pageSize=50"
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
        time.sleep(3)

        print("Intentando seleccionar 'All Results'...")
        try:
            boton_todo = espera.until(EC.element_to_be_clickable((By.ID, "allResults")))
            navegador.execute_script("arguments[0].click();", boton_todo)
            print("Botón 'All Results' clickeado correctamente.")
            time.sleep(5)
        except:
            print("No se encontró el botón 'All Results'.")

        print("Clickeando el botón 'Download'...")
        try:
            boton_descarga = espera.until(EC.element_to_be_clickable((By.CLASS_NAME, "downloadBtn")))
            navegador.execute_script("arguments[0].click();", boton_descarga)
            print("Botón 'Download' clickeado correctamente.")
        except:
            print("No se encontró el botón 'Download'.")

        # Esperar que aparezca el enlace "Download now!"
        print("Esperando hasta 2 minutos a que aparezca el enlace 'Download now!'...")
        try:
            enlace_descarga = WebDriverWait(navegador, 120).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a[download='acm.bib']"))
            )
            navegador.execute_script("arguments[0].click();", enlace_descarga)
            print("✅ Clic en 'Download now!' realizado correctamente.")
        except:
            print("⚠️ No se encontró el enlace 'Download now!' después de 2 minutos. Puede requerir intervención manual.")


        # Esperar un momento para que se complete la descarga
        print("Esperando que finalice la descarga...")
        time.sleep(15)

        print("Verificando si el archivo .bib fue descargado...")
        archivos_bib = sorted(Path(carpeta_descargas).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not archivos_bib:
            print("❌ No se encontró ningún archivo .bib en la carpeta de descargas.")
        else:
            archivo_reciente = archivos_bib[0]
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_final = carpeta_salida / f"acm_completo_{marca_tiempo}.bib"
            shutil.move(str(archivo_reciente), ruta_final)
            print(f"✅ Archivo BibTeX guardado en: {ruta_final}")

    except Exception as error:
        print(f"Error durante la extracción: {error}")


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
        carpeta_descargas = str(Path("downloads").resolve())
        ruta_perfil = r"C:\Users\hulbe\ChromeProfiles\scraper_profile"

        opciones = configurar_navegador(carpeta_descargas, ruta_perfil)

        print("Iniciando Chrome en modo sigiloso con perfil guardado...")
        navegador = uc.Chrome(options=opciones)
        espera = WebDriverWait(navegador, 25)

        extraer_todo_acm_bibtex(navegador, espera, carpeta_descargas)

        input("Presiona Enter para cerrar el navegador...")
        navegador.quit()

    except ValueError:
        print("Error en la entrada. Asegúrate de escribir correctamente la palabra clave.")
