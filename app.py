from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import time
import pandas as pd
from utils.unir_bib import main
from utils.analizador_palabras_clave import main_analizador_palabras_clave
from utils.agrupamiento.ejecutar_todo import main_analizador_dendrogramas
# from utils.visuals_req5 import main_visuals_req5
from utils.comparar_articulos import cargar_bib, comparar_articulos
from scraper.ieee_scraper import extraer_ieee_bibtex_chrome
from scraper.acm_scraper import extraer_acm_bibtex, configurar_navegador
from selenium.webdriver.support.ui import WebDriverWait
import undetected_chromedriver as uc
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

@app.route('/')
def index():
    return render_template('index.html')

# --- Scraper ACM ---
@app.route('/scraper/acm', methods=['POST'])
def scraper_acm():
    try:
        data = request.get_json()
        start = int(data.get('start', 0))
        count = int(data.get('count', 1))

        # Configura las rutas 
        carpeta_descargas = str(Path("downloads").resolve())
        ruta_perfil = str(Path.home() / "ChromeProfiles" / "scraper_profile")
        Path(carpeta_descargas).mkdir(parents=True, exist_ok=True)

        # Opciones navegador
        opciones = configurar_navegador(carpeta_descargas, ruta_perfil)
        navegador = uc.Chrome(options=opciones, version_main=141)
        espera = WebDriverWait(navegador, 20)

        for i in range(start, start + count):
            extraer_acm_bibtex(
                navegador, espera, carpeta_descargas,
                pagina_inicio=i,
                primera_pagina=(i == start),
                ultima_pagina=(i == start + count - 1)
            )
            time.sleep(3)

        navegador.quit()
        return jsonify({
            'status': 'success',
            'message': f'Scraper ACM completado. Procesadas {count} páginas.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- Scraper IEEE ---
@app.route('/scraper/ieee', methods=['POST'])
def scraper_ieee():
    try:
        data = request.get_json()
        start = int(data.get('start', 0))
        count = int(data.get('count', 1))

        # Configura las rutas 
        carpeta_descargas = str(Path("downloads").resolve())
        ruta_perfil = str(Path.home() / "ChromeProfiles" / "scraper_profile")
        Path(carpeta_descargas).mkdir(parents=True, exist_ok=True)

        # Opciones navegador
        opciones = configurar_navegador(carpeta_descargas, ruta_perfil)
        navegador = uc.Chrome(options=opciones, version_main=141)
        espera = WebDriverWait(navegador, 20)
        
        for i in range(start, start + count):
            extraer_ieee_bibtex_chrome(
                navegador, espera, carpeta_descargas,
                pagina=i
            )
            time.sleep(3)

        navegador.quit()        
        return jsonify({
            'status': 'success',
            'message': f'Scraper IEEE completado. Procesadas {count} páginas.'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- Merge Bib ---
@app.route('/merge', methods=['POST'])
def merge_bib():
    try:
        output_dir = "static/data/processed/"
        main()
        csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        csv_data = {}
        for csv_file in csv_files:
            file_path = os.path.join(output_dir, csv_file)
            df = pd.read_csv(file_path)
            csv_data[csv_file] = {
                "table": df.to_html(classes='table table-striped', index=False),
                "url": url_for('static', filename=f'data/processed/{csv_file}')
            }

        return jsonify({
            'status': 'success',
            'message': 'Archivos BibTeX fusionados correctamente.',
            'csv_data': csv_data   # <- Agrega esto
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- Comparar Artículos ---
@app.route('/compare', methods=['POST'])
def compare_articles():
    try:
        data = request.get_json()
        ids = data.get('ids', '')
        
        if not ids.strip():
            return jsonify({
                'status': 'error',
                'message': 'Debe ingresar al menos un ID.'
            }), 400
        
        # Cargar abstracts
        abstracts = cargar_bib()
        # Obtener lista de IDs
        ids_list = [id_.strip() for id_ in ids.split(",") if id_.strip()]
        
        # Llamada correcta pasando abstracts e IDs
        result = comparar_articulos(abstracts, ids_list)
        
        return jsonify({
            'status': 'success',
            'message': 'Comparación completada exitosamente.',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- Keywords Analyzer ---
@app.route('/keywords', methods=['POST'])
def analyze_keywords():
    try:
        main_analizador_palabras_clave()
        
        output_dir = "static/salidas/analizador_palabras_clave"
        images = [f for f in os.listdir(output_dir) if f.endswith(".png")]

        image_urls = [url_for('static', filename=f'salidas/analizador_palabras_clave/{img}') for img in images]

        return jsonify({
            'status': 'success',
            'message': 'Análisis de keywords completado.',
            'images': image_urls
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- Dendrograms Analyzer ---
@app.route('/dendrograms', methods=['POST'])
def analyze_dendrograms():
    try:
        main_analizador_dendrogramas()
        
        output_dir = "static/salidas/agrupamiento_y_dendrogramas"
        images = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

        image_urls = [url_for('static', filename=f'salidas/agrupamiento_y_dendrogramas/{img}') for img in images]

        # Leer CSVs
        csv_data = {}
        for csv_file in csv_files:
            file_path = os.path.join(output_dir, csv_file)
            df = pd.read_csv(file_path)
            csv_data[csv_file] = df.to_html(classes='table table-striped', index=False)
        
        return jsonify({
            'status': 'success',
            'message': 'Análisis de dendrogramas completado.',
            'images': image_urls,
            'csv_data': csv_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/list_bib_files')
def list_bib_files():
    base_dir = 'static/data/raw/'
    bib_info = {}
    for subdir in ['ACM', 'IEEE']:
        folder = os.path.join(base_dir, subdir)
        if os.path.exists(folder):
            bib_info[subdir] = [f for f in os.listdir(folder) if f.endswith('.bib')]
        else:
            bib_info[subdir] = []
    return jsonify({'files': bib_info})


if __name__ == '__main__':
    # Crear carpetas necesarias
    os.makedirs('static/salidas/analizador_palabras_clave', exist_ok=True)
    os.makedirs('static/salidas/agrupamiento_y_dendrogramas', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
