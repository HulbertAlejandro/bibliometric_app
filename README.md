# 📊 Bibliometric App

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Aplicación para la extracción automatizada y análisis bibliométrico de publicaciones académicas obtenidas desde fuentes digitales**

---

## Tabla de Contenidos

- [Introducción](#introducción)
- [Características Principales](#características-principales)
- [Fuentes de Información](#fuentes-de-información)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Contribución](#contribución)
- [Estado del Proyecto](#estado-del-proyecto)
- [Autores](#autores)
- [Licencia](#licencia)

---

## Introducción

La **bibliometría** explora y analiza grandes volúmenes de datos científicos utilizando métodos cuantitativos y cualitativos para establecer relaciones, inferencias y visualizaciones en publicaciones académicas. Algunos indicadores analizados incluyen:

- Productividad de autores
- Índices de impacto
- Distribución geográfica
- Tópicos y colaboración científica

---

## Características Principales

- Extracción y unificación automatizada de registros bibliográficos desde bases como ACM, SAGE y ScienceDirect.
- Implementación de algoritmos de similitud textual (clásicos + IA).
- Análisis de frecuencia de conceptos en abstracts y keywords.
- Agrupamiento jerárquico (clustering) y visualización (dendrogramas, mapas de calor, nubes de palabras, líneas temporales).
- Exportación de visualizaciones a PDF.
- Modularidad y facilidad de uso.

---

## Fuentes de Información

La aplicación soporta **ACM, SAGE, ScienceDirect** (además de bases institucionales de la Universidad del Quindío). Se aceptan formatos de exportación: **RIS, BibTeX, CSV, texto plano**.

**Dominio objetivo:** Inteligencia Artificial Generativa  
**Cadena de búsqueda:** “generative artificial intelligence”

---

## Estructura del Proyecto
- `/scraper` → Scripts para extracción automática de datos (actualmente IEEE y ACM soportados).
- `/utils` → Utilidades para procesamiento y limpieza de datos bibliográficos.
- `/data/processed` → Datos unificados y procesados listos para análisis.
- `main.py` → Script principal, punto de entrada del proyecto por consola.
- `app.py` → Script principal, punto de entrada del proyecto por web.
- `requerimientos.txt` → Dependencias del entorno Python.

---

## Instalación

````
git clone https://github.com/HulbertAlejandro/bibliometric_app.git
cd bibliometric_app
pip install -r requerimientos.txt
````

---

## Uso Rápido
````
python main.py
````

## Uso Con Interface Web
````
python app.py
````

---

## Estado del Proyecto

- [x] Extracción y unión básica de datos (IEEE y ACM)
- [x] Automatización de descarga de todas las bases requeridas
- [x] Implementación completa de algoritmos de similitud y clustering
- [x] Sistema de visualización y exportación PDF
- [x] Documentación final por requerimiento

---

## Autores

- **Hulbert Alejandro Arango** ([HulbertAlejandro](https://github.com/HulbertAlejandro))
- **Juan Esteban Cardona** ([iamjuaness](https://github.com/iamjuaness))
- **Mauricio Ríos** ([mauro-2002](https://github.com/mauro-2002))

---

## Licencia

Distribuido bajo Licencia MIT.