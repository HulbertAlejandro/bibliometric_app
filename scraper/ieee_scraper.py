# ieee_scraper_chrome.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from pathlib import Path
import shutil
import time
import os

def configurar_chrome(carpeta_descargas: str, ruta_perfil: str | None = None):
    """
    Configura Chrome (undetected-chromedriver) con preferencias de descarga y perfil.
    """
    opciones = uc.ChromeOptions()
    prefs = {
        "download.default_directory": carpeta_descargas,  # directorio absoluto
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    opciones.add_experimental_option("prefs", prefs)
    opciones.add_argument("--start-maximized")
    opciones.add_argument("--disable-popup-blocking")
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-blink-features=AutomationControlled")

    # Reutilizar un perfil (opcional pero útil si hay captcha o consentimiento previo)
    if ruta_perfil:
        # Ejemplo Windows: r"C:\Users\TUUSUARIO\AppData\Local\Google\Chrome\User Data"
        opciones.add_argument(rf"--user-data-dir={ruta_perfil}")
        # Cambia "Default" por "Profile 1" si corresponde a tu perfil
        opciones.add_argument("--profile-directory=Default")

    return opciones

def extraer_ieee_bibtex_chrome(navegador, espera, carpeta_descargas: str, pagina: int):
    """
    Abre IEEE Xplore en Chrome, acepta cookies, selecciona todos, exporta citaciones BibTeX con Abstract,
    y mueve el .bib a data/raw/IEEE con timestamp.
    """
    carpeta_salida = Path("data/raw/IEEE")
    Path(carpeta_descargas).mkdir(parents=True, exist_ok=True)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    url = (
        "https://ieeexplore.ieee.org/search/searchresult.jsp"
        "?newsearch=true&queryText=generative+artificial+intelligence"
        "&highlight=true&returnType=SEARCH&matchPubs=true"
        f"&rowsPerPage=100&pageNumber={pagina}&returnFacets=ALL"
    )
    navegador.get(url)

    try:
        # Aceptar cookies si aparece el botón (puede variar)
        print("Esperando para aceptar las cookies...")
        try:
            boton_cookies = espera.until(
                EC.element_to_be_clickable((
                    By.CSS_SELECTOR,
                    "button.osano-cm-accept-all.osano-cm-buttons__button.osano-cm-button.osano-cm-button--type_accept"
                ))
            )
            boton_cookies.click()
            print("Cookies aceptadas")
            time.sleep(2)
        except Exception:
            print("No se encontró el botón de cookies (posiblemente ya aceptadas).")

        # Seleccionar todos los resultados
        print("Seleccionando todos los resultados...")
        casilla_todo = espera.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "input.xpl-checkbox-default.results-actions-selectall-checkbox"
            ))
        )
        navegador.execute_script("arguments[0].click();", casilla_todo)
        time.sleep(2)

        # Abrir modal de Export
        print("Abriendo el modal de exportación...")
        boton_exportar = espera.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//button[@class='xpl-btn-primary' and normalize-space(text())='Export']"
            ))
        )
        navegador.execute_script("arguments[0].click();", boton_exportar)
        time.sleep(2)

        # Ir a la pestaña Citations
        print("Accediendo al modal de 'Citations'...")
        pestaña_citations = espera.until(
            EC.element_to_be_clickable((
                By.XPATH,
                "//a[@class='nav-link' and normalize-space(text())='Citations']"
            ))
        )
        navegador.execute_script("arguments[0].click();", pestaña_citations)
        time.sleep(2)

        # Seleccionar BibTeX
        print("Seleccionando formato BibTeX...")
        bibtex_input = espera.until(
            EC.presence_of_element_located((
                By.XPATH,
                '//label[.//span[normalize-space()="BibTeX"]]/input'
            ))
        )
        navegador.execute_script("""
            arguments[0].checked = true;
            arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
        """, bibtex_input)
        time.sleep(1)

        # Seleccionar "Citation and Abstract"
        print("Seleccionando 'Citation and Abstract'...")
        citation_input = espera.until(
            EC.presence_of_element_located((
                By.XPATH,
                '//label[.//span[normalize-space()="Citation and Abstract"]]/input'
            ))
        )
        navegador.execute_script("""
            arguments[0].checked = true;
            arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
        """, citation_input)
        time.sleep(2)

        # Descargar
        print("Clickeando 'Descargar'...")
        boton_descarga = espera.until(
            EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                "button.stats-SearchResults_Citation_Download.xpl-btn-primary"
            ))
        )
        navegador.execute_script("arguments[0].click();", boton_descarga)
        print("Descarga solicitada.")
        time.sleep(7)  # espera por la descarga

        # Mover el archivo .bib
        archivos_bib = sorted(Path(carpeta_descargas).glob("*.bib"), key=os.path.getmtime, reverse=True)
        if not archivos_bib:
            print("No hay archivo .bib encontrado en descargas")
            return

        archivo_reciente = archivos_bib[0]
        marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_final = carpeta_salida / f"ieee_scraped_{marca_tiempo}.bib"
        shutil.move(str(archivo_reciente), ruta_final)

        print(f"BibTeX guardado en: {ruta_final}")

    except Exception as e:
        print(f"Error en el scraping de página {pagina}: {e}")

def ejecutar():
    try:
        inicio = int(input("Página de Inicio (e.g. 0): "))
        cantidad = int(input("¿Cuántas páginas para el scrape?: "))

        carpeta_descargas = str(Path("downloads").resolve())
        # Ajusta esta ruta a tu perfil real de Chrome si quieres reutilizar sesión/cookies/captcha
        ruta_perfil = str(Path.home() / "ChromeProfiles" / "scraper_profile")

        opciones = configurar_chrome(carpeta_descargas, ruta_perfil)
        print("Iniciando Chrome en modo sigiloso con perfil (si aplica)...")
        navegador = uc.Chrome(options=opciones, version_main=141)
        espera = WebDriverWait(navegador, 20)

        for i in range(inicio, inicio + cantidad):
            print(f"\n>>> Scraping página {i}")
            extraer_ieee_bibtex_chrome(navegador, espera, carpeta_descargas, pagina=i)
            time.sleep(3)

        input("Presiona Enter para cerrar el navegador...")
        navegador.quit()

    except ValueError:
        print("Entrada no válida")

if __name__ == "__main__":
    ejecutar()
