from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualización y exportación
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, 
                                PageBreak, Table, TableStyle, Frame, PageTemplate,
                                BaseDocTemplate, NextPageTemplate)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# Parseo y procesamiento
import bibtexparser
import pycountry

# NLP
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('punkt', quiet=True)
    STOPWORDS_AVAILABLE = True
except ImportError:
    STOPWORDS_AVAILABLE = False


# ==================== CONFIGURACIÓN ====================

COLOR_PALETTE = {
    'primary': '#308113',    # Verde principal (oscuro, elegante)
    'secondary': '#10A566',  # Verde secundario (brillante/acento natural)
    'accent': '#202824',     # Amarillo suave (contraste cálido, combina con verdes)
    'success': '#43B55C',    # Verde éxito (para "positivo" en gráficas)
    'warning': '#FFA556',    # Naranja pastel (suave, para advertencias o tendencias)
    'danger': '#D94352',     # Rojo combinado (alertas o valores críticos)
    'dark': '#202824',       # Verde pino oscuro casi negro (para backgrounds, textos)
    'light': '#F5FFF4',      # Verde muy claro/blanco (fondo, suavidad)
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


# ==================== FUNCIONES DE CARGA ====================

def cargar_bib(bib_path: str | Path) -> list[dict]:
    bib_path = Path(bib_path)
    with open(bib_path, "r", encoding="utf-8", errors="ignore") as f:
        db = bibtexparser.load(f)
    return db.entries


def normalize_country(text: str) -> str | None:
    if not text:
        return None
    t = text.strip()

    alias = {
        "USA": "United States", "U.S.A.": "United States", "US": "United States",
        "UK": "United Kingdom", "U.K.": "United Kingdom", "England": "United Kingdom",
        "JPN": "Japan, Japan",
        "MYS": "Malaysia, Malaysia",
        "PRT": "Portugal, Portugal",
        "CHN": "China, China",
        "ITA": "Italy, Italy",
        "GBR": "United Kingdom, United Kingdom",
        "DEU": "Germany, Germany",
        "CRI": "Costa Rica, Costa Rica",
        "ARG": "Argentina, Argentina",
        "BRA": "Brazil, Brazil",
        "COL": "Colombia, Colombia",
        "MEX": "Mexico, Mexico",
        "VEN": "Venezuela, Venezuela",
        "FIN": "Finland, Finland",
        "RUS": "Russia, Russia",
        "POL": "Poland, Poland",
        "CAN": "Canada, Canada",
        "GRC": "Greece, Greece",
        "NLD": "Netherlands, Netherlands",
        "ESP": "Spain, Spain",
        "IND": "India, India",
        "AUT": "Austria, Austria",
        "SWE": "Sweden, Sweden",
        "DNK": "Denmark, Denmark",
        "IRL": "Ireland, Ireland",
        "CHE": "Switzerland, Switzerland",
        "THA": "Thailand, Thailand",
        "BEL": "Belgium, Belgium",
        "UNK": "UNKNOWN"
    }

    if t in alias:
        t = alias[t]

    for country in pycountry.countries:
        names = {country.name}
        if hasattr(country, "official_name"):
            names.add(getattr(country, "official_name"))
        if t.lower() in {n.lower() for n in names}:
            return country.alpha_3

    for country in pycountry.countries:
        candidates = {country.name}
        if hasattr(country, "official_name"):
            candidates.add(getattr(country, "official_name"))
        for cand in candidates:
            if cand and len(cand) > 3 and cand.lower() in t.lower():
                return country.alpha_3

    return None


def extraer_pais_primer_autor(entry: dict) -> str:
    priority_fields = ["affiliation", "affiliations", "author+affiliation", "author_affiliation"]

    for field in priority_fields:
        aff = entry.get(field) or ""
        if aff:
            first_aff = re.split(r"[;,]| and | AND ", aff)[0].strip()
            iso3 = normalize_country(first_aff)
            if iso3:
                return iso3

    for field in ["address", "location", "publisher"]:
        address = entry.get(field, "")
        iso3 = normalize_country(address)
        if iso3:
            return iso3

    return "UNK"


def limpiar_texto_para_wordcloud(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    if STOPWORDS_AVAILABLE:
        stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
        stop_words.update(['using', 'based', 'approach', 'method', 'algorithm', 'system', 
                          'paper', 'study', 'analysis', 'research', 'data', 'results'])
        words = word_tokenize(text)
        text = ' '.join([word for word in words if word not in stop_words and len(word) > 2])

    return text


def to_dataframe(entries: list[dict]) -> pd.DataFrame:
    records = []
    for e in entries:
        year = e.get("year")
        try:
            year_match = re.search(r'(19|20)\d{2}', str(year))
            year = int(year_match.group()) if year_match else None
        except Exception:
            year = None

        venue = (e.get("journal") or e.get("booktitle") or 
                e.get("publisher") or e.get("series") or "Otros")
        venue = re.sub(r'\{|\}', '', venue).strip()
        pub_type = e.get("ENTRYTYPE", "article").lower()

        records.append({
            "title": e.get("title", ""),
            "authors": e.get("author", ""),
            "abstract": e.get("abstract", ""),
            "keywords": e.get("keywords", ""),
            "year": year,
            "venue": venue,
            "pub_type": pub_type,
            "first_author_country": extraer_pais_primer_autor(e),
        })

    df = pd.DataFrame.from_records(records)
    df = df[(df['year'] >= 1990) & (df['year'] <= 2030)]
    return df


# ==================== VISUALIZACIONES ====================

def construir_mapa_de_calor_geoespacial(df: pd.DataFrame, out_png: Path) -> None:
    counts = df["first_author_country"].value_counts().reset_index()
    counts.columns = ["iso_alpha", "count"]
    counts = counts[counts["iso_alpha"] != "UNK"]

    if counts.empty:
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        ax.text(0.5, 0.5, 'Distribución Geográfica\n(Sin datos)', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax.axis('off')
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return

    fig = px.choropleth(
        counts,
        locations="iso_alpha",
        color="count",
        color_continuous_scale=[[0, COLOR_PALETTE['light']], 
                               [0.5, COLOR_PALETTE['secondary']], 
                               [1, COLOR_PALETTE['primary']]],
        projection="natural earth",
        title="<b>Distribución Geográfica de Publicaciones Científicas</b>",
        labels={"count": "Publicaciones", "iso_alpha": "País"}
    )

    fig.update_layout(
        font=dict(family="Arial", size=13),
        title_font_size=20,
        title_font_color=COLOR_PALETTE['primary'],
        title_x=0.5,
        width=1400,
        height=800,
        paper_bgcolor='white',
        geo=dict(showframe=True, showcoastlines=True)
    )

    fig.write_image(str(out_png), width=1400, height=800, scale=2)


def construir_wordcloud(df: pd.DataFrame, out_png: Path) -> None:
    corpus = []
    for _, row in df.iterrows():
        if isinstance(row["abstract"], str):
            corpus.append(limpiar_texto_para_wordcloud(row["abstract"]))
        if isinstance(row["keywords"], str):
            corpus.append(limpiar_texto_para_wordcloud(row["keywords"]))

    text = " ".join(corpus) or "No data"

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['warning']]
        return colors[random_state.randint(0, len(colors)-1)] if random_state else colors[0]

    wc = WordCloud(
        width=1800, height=1000, background_color="white",
        max_words=120, color_func=color_func
    )

    fig, ax = plt.subplots(figsize=(18, 10), facecolor='white')
    ax.imshow(wc.generate(text), interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Términos Más Frecuentes', fontsize=22, fontweight='bold', 
                color=COLOR_PALETTE['primary'], pad=25)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def construir_linea_de_tiempo(df: pd.DataFrame, out_png: Path) -> None:
    df_valid = df.dropna(subset=['year'])
    yearly_counts = df_valid.groupby('year').size().reset_index(name='count')
    top_venues = df_valid['venue'].value_counts().head(5).index.tolist()
    venue_yearly = df_valid[df_valid['venue'].isin(top_venues)].groupby(['year', 'venue']).size().reset_index(name='count')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("<b>Evolución Total</b>", "<b>Por Tipo</b>", 
                       "<b>Top 5 Venues</b>", "<b>Tendencia</b>"),
        specs=[[{}, {"type": "pie"}], [{}, {}]],
        vertical_spacing=0.12
    )

    # 1. Línea temporal
    fig.add_trace(
        go.Scatter(x=yearly_counts['year'], y=yearly_counts['count'],
                  mode='lines+markers', name='Publicaciones',
                  line=dict(color=COLOR_PALETTE['primary'], width=3),
                  fill='tozeroy'),
        row=1, col=1
    )

    # 2. Pie chart
    pub_types = df_valid['pub_type'].value_counts()
    fig.add_trace(
        go.Pie(labels=pub_types.index, values=pub_types.values, hole=0.4),
        row=1, col=2
    )

    # 3. Venues
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']]
    for i, venue in enumerate(top_venues):
        data = venue_yearly[venue_yearly['venue'] == venue]
        fig.add_trace(
            go.Scatter(x=data['year'], y=data['count'], name=venue[:30],
                      stackgroup='one', line=dict(color=colors[i % len(colors)])),
            row=2, col=1
        )

    # 4. Tendencia
    if len(yearly_counts) > 1:
        z = np.polyfit(yearly_counts['year'], yearly_counts['count'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=yearly_counts['year'],
                y=yearly_counts['count'],
                mode='markers',
                name='Datos Reales',
                marker=dict(color='#D94352', size=10)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=yearly_counts['year'],
                y=p(yearly_counts['year']),
                mode='lines',
                name='Tendencia Linear',
                line=dict(color='#43B55C', width=3, dash='dash')
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=900, width=1600,
        title_text="<b>Análisis Temporal</b>",
        title_x=0.5, font=dict(family="Arial", size=11),
        paper_bgcolor='white'
    )

    fig.write_image(str(out_png), width=1600, height=900, scale=2)


# ==================== ESTADÍSTICAS ====================

def crear_stats_resumidas(df: pd.DataFrame) -> dict:
    df_country = df[df['first_author_country'] != 'UNK']

    return {
        'total_publications': len(df),
        'year_range': f"{int(df['year'].min())} - {int(df['year'].max())}",
        'countries_count': df_country['first_author_country'].nunique(),
        'venues_count': df['venue'].nunique(),
        'top_country': df_country['first_author_country'].mode().iloc[0] if len(df_country) > 0 else 'N/A',
        'top_venue': df['venue'].mode().iloc[0],
        'publications_per_year': f"{df.groupby('year').size().mean():.1f}",
        'peak_year': int(df['year'].mode().iloc[0]),
        'std_per_year': f"{df.groupby('year').size().std():.1f}",
    }


# ==================== PDF PROFESIONAL - CORREGIDO ====================

def crear_portada(canvas_obj, doc):
    """Crea portada del PDF."""
    canvas_obj.saveState()

    canvas_obj.setFillColor(HexColor(COLOR_PALETTE['primary']))
    canvas_obj.rect(0, 0, A4[0], A4[1], fill=True, stroke=False)

    canvas_obj.setFillColor(HexColor(COLOR_PALETTE['accent']))
    canvas_obj.rect(0, A4[1] - 150, A4[0], 150, fill=True, stroke=False)

    canvas_obj.setFont("Helvetica-Bold", 32)
    canvas_obj.setFillColor(white)
    canvas_obj.drawCentredString(A4[0] / 2.0, A4[1] - 80, "ANÁLISIS BIBLIOMÉTRICO")

    canvas_obj.setFont("Helvetica", 18)
    canvas_obj.drawCentredString(A4[0] / 2.0, A4[1] - 110, "Reporte de Producción Científica")

    canvas_obj.setStrokeColor(white)
    canvas_obj.setLineWidth(3)
    canvas_obj.line(100, A4[1] / 2.0 + 50, A4[0] - 100, A4[1] / 2.0 + 50)

    canvas_obj.setFont("Helvetica-Bold", 14)
    canvas_obj.setFillColor(HexColor(COLOR_PALETTE['accent']))
    canvas_obj.drawCentredString(A4[0] / 2.0, A4[1] / 2.0 + 10, "Sistema de Análisis y Visualización")

    canvas_obj.setFont("Helvetica", 12)
    canvas_obj.setFillColor(white)
    canvas_obj.drawCentredString(A4[0] / 2.0, A4[1] / 2.0 - 20, "Universidad del Quindío")
    canvas_obj.drawCentredString(A4[0] / 2.0, A4[1] / 2.0 - 40, "Facultad de Ingeniería")

    canvas_obj.setFont("Helvetica-Bold", 11)
    fecha_actual = datetime.now().strftime("%B %Y")
    canvas_obj.drawCentredString(A4[0] / 2.0, 100, fecha_actual.upper())

    canvas_obj.restoreState()


def crear_header(canvas_obj, doc):
    """Header de cada página."""
    canvas_obj.saveState()

    canvas_obj.setFillColor(HexColor(COLOR_PALETTE['primary']))
    canvas_obj.rect(0, A4[1] - 60, A4[0], 60, fill=True, stroke=False)

    canvas_obj.setFillColor(HexColor(COLOR_PALETTE['accent']))
    canvas_obj.rect(0, A4[1] - 65, A4[0], 5, fill=True, stroke=False)

    canvas_obj.setFont("Helvetica-Bold", 16)
    canvas_obj.setFillColor(white)
    canvas_obj.drawString(50, A4[1] - 40, "Análisis Bibliométrico")

    canvas_obj.setFont("Helvetica", 9)
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    canvas_obj.drawRightString(A4[0] - 50, A4[1] - 40, f"Generado: {fecha_actual}")

    canvas_obj.restoreState()


def crear_footer(canvas_obj, doc):
    """Footer de cada página."""
    canvas_obj.saveState()

    canvas_obj.setStrokeColor(HexColor(COLOR_PALETTE['light']))
    canvas_obj.setLineWidth(2)
    canvas_obj.line(50, 50, A4[0] - 50, 50)

    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.setFillColor(HexColor(COLOR_PALETTE['dark']))
    page_num = canvas_obj.getPageNumber()
    canvas_obj.drawCentredString(A4[0] / 2.0, 35, f"Página {page_num}")

    canvas_obj.setFont("Helvetica-Oblique", 7)
    canvas_obj.drawString(50, 35, "Sistema de Análisis Bibliográfico")
    canvas_obj.drawRightString(A4[0] - 50, 35, "Universidad del Quindío")

    canvas_obj.restoreState()


def on_page_portada(canvas_obj, doc):
    """Callback para la portada."""
    crear_portada(canvas_obj, doc)


def on_page_contenido(canvas_obj, doc):
    """Callback para páginas de contenido."""
    crear_header(canvas_obj, doc)
    crear_footer(canvas_obj, doc)


def exportar_pdf(images: list[Path], out_pdf: Path, stats: dict) -> None:
    """Exporta PDF con plantilla profesional - CORREGIDO."""

    doc = BaseDocTemplate(
        str(out_pdf),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=80,
        bottomMargin=70
    )

    # Frames
    frame_portada = Frame(0, 0, A4[0], A4[1], id='frame_portada',
                         leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    frame_contenido = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                           id='frame_contenido')

    # Templates
    template_portada = PageTemplate(id='portada', frames=[frame_portada],
                                   onPage=on_page_portada, pagesize=A4)

    template_contenido = PageTemplate(id='contenido', frames=[frame_contenido],
                                     onPage=on_page_contenido, pagesize=A4)

    doc.addPageTemplates([template_portada, template_contenido])

    # Estilos
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                fontSize=26, textColor=HexColor(COLOR_PALETTE['primary']),
                                spaceAfter=20, alignment=TA_CENTER, fontName='Helvetica-Bold')

    subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Heading2'],
                                   fontSize=16, textColor=HexColor(COLOR_PALETTE['primary']),
                                   spaceAfter=12, fontName='Helvetica-Bold')

    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading3'],
                                  fontSize=13, textColor=HexColor(COLOR_PALETTE['secondary']),
                                  spaceAfter=8, fontName='Helvetica-Bold')

    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
                                 fontSize=10, textColor=HexColor(COLOR_PALETTE['dark']),
                                 alignment=TA_JUSTIFY, spaceAfter=8, leading=14)

    story = []

    # Portada
    story.append(NextPageTemplate('contenido'))
    story.append(PageBreak())

    # Resumen
    story.append(Paragraph("RESUMEN EJECUTIVO", title_style))
    story.append(Spacer(1, 10))

    resumen = f"""
    Este reporte presenta un análisis bibliométrico completo de <b>{stats['total_publications']}</b> publicaciones 
    científicas del período <b>{stats['year_range']}</b>. Incluye distribución geográfica, evolución temporal 
    y análisis de contenido mediante procesamiento de lenguaje natural.
    """
    story.append(Paragraph(resumen, normal_style))
    story.append(Spacer(1, 15))

    # Estadísticas
    story.append(Paragraph("1. Estadísticas Generales", subtitle_style))
    story.append(Spacer(1, 10))

    stats_data = [
        ['Métrica', 'Valor'],
        ['Total Publicaciones', str(stats['total_publications'])],
        ['Rango Temporal', stats['year_range']],
        ['Países', str(stats['countries_count'])],
        ['Venues', str(stats['venues_count'])],
        ['País Top', stats['top_country']],
        ['Promedio/Año', stats['publications_per_year']],
        ['Año Pico', str(stats['peak_year'])],
        ['Desviación', stats.get('std_per_year', 'N/A')],
    ]

    table = Table(stats_data, colWidths=[8*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor(COLOR_PALETTE['primary'])),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor(COLOR_PALETTE['light'])]),
        ('BOX', (0, 0), (-1, -1), 1.5, HexColor(COLOR_PALETTE['primary'])),
        ('GRID', (0, 1), (-1, -1), 0.5, HexColor('#E5E7EB'))
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    venue_text = f"<b>Venue Principal:</b> {stats['top_venue'][:100]}"
    story.append(Paragraph(venue_text, normal_style))
    story.append(PageBreak())

    # Visualizaciones
    titulos = [
        ("2. Distribución Geográfica", "Mapa de publicaciones por país del primer autor."),
        ("3. Análisis de Contenido", "Nube de palabras de abstracts y keywords."),
        ("4. Análisis Temporal", "Evolución temporal con múltiples perspectivas.")
    ]

    for i, (img_path, (titulo, desc)) in enumerate(zip(images, titulos)):
        if img_path.exists():
            story.append(Paragraph(titulo, subtitle_style))
            story.append(Spacer(1, 8))
            story.append(Paragraph(desc, normal_style))
            story.append(Spacer(1, 12))

            try:
                img = Image(str(img_path))
                aspect = img.imageWidth / img.imageHeight
                if aspect > 1.37:
                    img.drawWidth = 480
                    img.drawHeight = 480 / aspect
                else:
                    img.drawHeight = 350
                    img.drawWidth = 350 * aspect
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 15))
            except Exception as e:
                story.append(Paragraph(f"<i>Error: {e}</i>", normal_style))

            if i < len(images) - 1:
                story.append(PageBreak())

    # Conclusiones
    story.append(PageBreak())
    story.append(Paragraph("5. Conclusiones", subtitle_style))
    story.append(Spacer(1, 10))

    conclusiones = f"""
    <b>Producción:</b> {stats['total_publications']} publicaciones ({stats['year_range']}), 
    promedio de {stats['publications_per_year']} por año.<br/><br/>
    <b>Geografía:</b> {stats['countries_count']} países representados, liderados por {stats['top_country']}.<br/><br/>
    <b>Diversidad:</b> {stats['venues_count']} venues únicos identificados.<br/><br/>
    <b>Tendencia:</b> Año pico en {stats['peak_year']}.
    """
    story.append(Paragraph(conclusiones, normal_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Nota Metodológica", section_style))
    nota = """
    Análisis generado con técnicas de bibliometría computacional y procesamiento de lenguaje natural.
    Herramientas: Python, pandas, plotly, matplotlib, nltk.
    """
    story.append(Paragraph(nota, normal_style))

    # Build
    try:
        doc.build(story)
        print(f"PDF generado: {out_pdf}")
    except Exception as e:
        print(f"Usando plantilla simple: {e}")
        SimpleDocTemplate(str(out_pdf), pagesize=A4).build(story)


# ==================== FUNCIÓN PRINCIPAL ====================

def run(bib_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    entries = cargar_bib(bib_path)
    df = to_dataframe(entries)
    print(f"{len(df)} entradas")

    heatmap_png = out_dir / "geo_heatmap.png"
    wordcloud_png = out_dir / "wordcloud.png" 
    timeline_png = out_dir / "timeline.png"
    pdf_path = out_dir / "reporte_profesional.pdf"

    print("  Mapa...")
    construir_mapa_de_calor_geoespacial(df, heatmap_png)

    print("  Wordcloud...")
    construir_wordcloud(df, wordcloud_png)

    print(" Timeline...")
    construir_linea_de_tiempo(df, timeline_png)

    print(" Stats...")
    stats = crear_stats_resumidas(df)

    print(" PDF...")
    exportar_pdf([heatmap_png, wordcloud_png, timeline_png], pdf_path, stats)

    # Reemplazar comas por punto y coma en todas las celdas de texto (str)
    df = df.applymap(lambda x: x.replace(",", ";") if isinstance(x, str) else x)


    df.to_csv(out_dir / "debug.csv", index=False)

    print("\n ¡Completado!")
    return {"pdf": str(pdf_path), "stats": stats}


def main_bibliometrica():
    bib_path = Path("static/data/processed/merged.bib")
    out_dir = Path("static/salidas/info_bibliometrica")

    if not bib_path.exists():
        print(f" Archivo no encontrado: {bib_path}")
        return

    print("="*70)
    print("  SISTEMA DE ANÁLISIS BIBLIOMÉTRICO")
    print("="*70)

    results = run(bib_path, out_dir)

    print("\n ESTADÍSTICAS:")
    for k, v in results['stats'].items():
        print(f"  • {k}: {v}")
    print("="*70)


if __name__ == "__main__":
    main_bibliometrica()