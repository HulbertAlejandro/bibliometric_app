# 📊 Bibliometric App

Este proyecto implementa una aplicación para la **extracción automatizada** y **análisis bibliométrico** de publicaciones académicas obtenidas desde fuentes digitales.

## 1. Introducción

La **bibliometría** permite explorar y analizar grandes volúmenes de datos científicos, usando métodos cuantitativos y cualitativos para establecer relaciones, inferencias y visualizaciones en publicaciones académicas. Se analizan indicadores como productividad de autores, índices de impacto, distribución geográfica, tópicos y colaboración científica.

## 2. Fuentes de Información

La app trabaja con bases de datos científicas como ACM, SAGE y ScienceDirect, disponibles en la Universidad del Quindío ([ver bases de datos](https://library.uniquindio.edu.co/databases)). Se soportan formatos de exportación como RIS, BibTeX, CSV y texto plano.

> **Dominio de conocimiento objetivo:** Inteligencia Artificial Generativa  
> **Cadena de búsqueda:** "generative artificial intelligence"

## 3. Propósito del Proyecto

Desarrollar algoritmos que permitan el análisis bibliométrico sobre el dominio de IA generativa usando datos extraídos desde las fuentes científicas, automatizando el flujo desde la búsqueda hasta el análisis y visualización.

## 4. Estructura del Proyecto

- `/scraper` → Scripts para extracción automática de datos (actualmente IEEE y ACM soportados).
- `/utils` → Utilidades para procesamiento y limpieza de datos bibliográficos.
- `/data/processed` → Datos unificados y procesados listos para análisis.
- `main.py` → Script principal, punto de entrada del proyecto.
- `requerimientos.txt` → Dependencias del entorno Python.

## 5. Funcionalidades y Requerimientos

### 5.1 Descarga y Unificación de Datos
- Automatización de descarga desde al menos dos bases de datos.
- Unificación automática de registros, eliminando duplicados.
- Exportación de archivos:
  - Uno con registros únicos y completos (autores, título, palabras clave, resumen, etc.)
  - Otro con registros eliminados por repetición.

### 5.2 Algoritmos de Similitud Textual
- Implementación de al menos **cuatro algoritmos clásicos** (por ejemplo, distancia de edición, vectorización).
- Implementación de **dos algoritmos con modelos de IA**.
- Comparación paso a paso, seleccionando artículos y analizando abstracts.

### 5.3 Análisis de Frecuencia de Conceptos
- Cálculo de frecuencia de términos asociados a la categoría *Concepts of Generative AI in Education*:
  - Generative models, Prompting, Machine learning, Multimodality, Fine-tuning, Training data, Algorithmic bias, Explainability, Transparency, Ethics, Privacy, Personalization, Human-AI interaction, AI literacy, Co-creation.
- Algoritmo para descubrir hasta 15 nuevas palabras en los abstracts y evaluar su precisión.

### 5.4 Agrupamiento Jerárquico
- Implementar **tres algoritmos de clustering jerárquico**.
- Representar resultados en dendrogramas, comparar coherencia de agrupamientos.

### 5.5 Visualizaciones Científicas
- **Mapa de calor** con distribución geográfica por autor principal.
- **Nube de palabras** dinámica sobre abstracts y keywords.
- **Línea temporal** de publicaciones por año y revista.
- **Exportación de visualizaciones a PDF**.

### 5.6 Despliegue y Documentación
- El proyecto se debe desplegar (Web/Local) y soportar con documentación técnica para cada requerimiento.

## 6. Instalación y Uso

```
git clone https://github.com/HulbertAlejandro/bibliometric_app.git
cd bibliometric_app
pip install -r requerimientos.txt
python main.py
```

## 7. Contribución

Las contribuciones son bienvenidas. Por favor, abre un *pull request* o sugerencia en la sección de *issues*.

## 8. Estado Actual/Pendiente

- [x] Extracción y unión básica de datos (IEEE y ACM)
- [ ] Automatización de descarga para todas las bases requeridas
- [ ] Implementación completa de algoritmos de similitud y clustering
- [ ] Sistema de visualización y exportación PDF
- [ ] Pruebas y documentación final por requerimiento

## 9. Autores

- Hulbert Alejandro Arango (@HulbertAlejandro)
- Juan Esteban Cardona (@iamjuaness)
- Mauricio Ríos (@mauro-2002)
