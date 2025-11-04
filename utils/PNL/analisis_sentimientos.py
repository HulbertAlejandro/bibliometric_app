"""
Análisis de Sentimientos con NLP Avanzado
Complejidad: O(n·m) donde n=documentos, m=tokens promedio
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Topic Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Descargar recursos de NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass


class AnalizadorDeSentimientos:
    """
    Analizador de sentimientos multi-método
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocesar_texto(self, text):
        """
        Preprocesa texto eliminando stopwords y puntuación
        Complejidad: O(m) donde m = número de tokens
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]
        return ' '.join(tokens)
    
    def vader_sentiment(self, text):
        """
        Análisis VADER - Optimizado para textos cortos y redes sociales
        Complejidad: O(m) donde m = número de palabras
        """
        if pd.isna(text) or text == "":
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0, 'label': 'neutral'}
        
        scores = self.vader_analyzer.polarity_scores(str(text))
        
        # Clasificación basada en compound score
        if scores['compound'] >= 0.05:
            label = 'positive'
        elif scores['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        scores['label'] = label
        return scores
    
    def textblob_sentiment(self, text):
        """
        Análisis TextBlob - Basado en polaridad y subjetividad
        Complejidad: O(m)
        """
        if pd.isna(text) or text == "":
            return {'polarity': 0, 'subjectivity': 0, 'label': 'neutral'}
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'label': label
        }
    
    def ensemble_sentiment(self, text):
        """
        Combina VADER y TextBlob para análisis robusto
        Complejidad: O(m)
        """
        vader = self.vader_sentiment(text)
        textblob = self.textblob_sentiment(text)
        
        # Pesos para cada método
        scores = {
            'vader_compound': vader['compound'],
            'textblob_polarity': textblob['polarity'],
            'vader_label': vader['label'],
            'textblob_label': textblob['label']
        }
        
        # Votación mayoritaria para label final
        labels = [vader['label'], textblob['label']]
        
        # Contar votos
        from collections import Counter
        vote = Counter(labels)
        final_label = vote.most_common(1)[0][0]
        
        scores['final_label'] = final_label
        scores['confidence'] = vote[final_label] / len(labels)
        
        return scores


class ModeloDeTopicos:
    """
    Modelado de tópicos con LDA
    """
    
    def __init__(self, n_topics=5, max_features=1000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
    
    def fit_lda(self, documents):
        """
        Entrena modelo LDA
        Complejidad: O(n·k·i) donde n=docs, k=topics, i=iteraciones
        """
        # Filtrar documentos vacíos
        documents = [str(doc) if pd.notna(doc) and str(doc).strip() else "empty" for doc in documents]
        
        # Vectorización TF
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=50,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        return self.lda_model, self.feature_names
    
    def get_topics(self, n_words=10):
        """
        Extrae palabras principales por tópico
        """
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic[top_indices]
            })
        return topics
    
    def predict_topics(self, documents):
        """
        Predice tópicos para nuevos documentos
        """
        # Filtrar documentos vacíos
        documents = [str(doc) if pd.notna(doc) and str(doc).strip() else "empty" for doc in documents]
        
        doc_term_matrix = self.vectorizer.transform(documents)
        topic_distributions = self.lda_model.transform(doc_term_matrix)
        
        # Tópico dominante por documento
        dominant_topics = topic_distributions.argmax(axis=1)
        
        return dominant_topics, topic_distributions


class AnalizadorDeTendenciasTemporales:
    """
    Análisis de tendencias temporales
    """
    
    def __init__(self):
        self.sentiment_analyzer = AnalizadorDeSentimientos()
    
    def _safe_text_preview(self, text, max_length=200):
        """
        Obtiene preview de texto manejando valores NaN y float
        Complejidad: O(1)
        """
        if pd.isna(text) or text is None:
            return ""
        
        text_str = str(text)
        
        # REEMPLAZAR COMAS POR PUNTO Y COMA
        text_str = text_str.replace(',', ';')
        
        # Limpiar espacios múltiples y saltos de línea
        text_str = ' '.join(text_str.split())
        
        return text_str[:max_length] if len(text_str) > max_length else text_str
    
    def analyze_temporal_trends(self, df, text_column='abstract', date_column='year'):
        """
        Analiza evolución temporal de sentimientos
        Complejidad: O(n·m) donde n=documentos, m=tokens
        """
        results = []
        
        total = len(df)
        print(f"Procesando {total} documentos...")
        
        for idx, row in df.iterrows():
            # Mostrar progreso cada 100 documentos
            if (idx + 1) % 100 == 0:
                print(f"  Procesados: {idx + 1}/{total} ({((idx + 1)/total)*100:.1f}%)")
            
            text = row.get(text_column, '')
            date = row.get(date_column, None)
            
            # Análisis de sentimiento
            sentiment = self.sentiment_analyzer.ensemble_sentiment(text)
            
            results.append({
                'index': idx,
                date_column: date,
                'text_preview': self._safe_text_preview(text, 200),
                **sentiment
            })
        
        print(f"Completado: {total}/{total} documentos procesados")
        return pd.DataFrame(results)
    
    def plot_sentiment_evolution(self, df_results, date_column='date', output_path='static/salidas/sentiment_analysis'):
        """
        Visualiza evolución temporal de sentimientos
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Verificar que hay datos de fecha válidos
        valid_dates = df_results[date_column].notna()
        if not valid_dates.any():
            print("No hay datos de fecha válidos para visualización temporal")
            return
        
        df_filtered = df_results[valid_dates].copy()
        
        # Agrupar por fecha y sentimiento
        sentiment_by_date = df_filtered.groupby([date_column, 'final_label']).size().unstack(fill_value=0)
        
        if sentiment_by_date.empty:
            print("No hay suficientes datos para generar visualizaciones")
            return
        
        # Gráfico de líneas
        plt.figure(figsize=(14, 6))
        sentiment_by_date.plot(kind='line', marker='o', linewidth=2, markersize=8)
        plt.title('Evolución Temporal de Sentimientos', fontsize=16, fontweight='bold')
        plt.xlabel('Año', fontsize=12)
        plt.ylabel('Número de Documentos', fontsize=12)
        plt.legend(title='Sentimiento', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_path}/sentiment_evolution.png', dpi=300)
        plt.close()
        
        # Gráfico de área apilada
        plt.figure(figsize=(14, 6))
        sentiment_by_date.plot(kind='area', stacked=True, alpha=0.7)
        plt.title('Distribución de Sentimientos por Año', fontsize=16, fontweight='bold')
        plt.xlabel('Año', fontsize=12)
        plt.ylabel('Proporción de Documentos', fontsize=12)
        plt.legend(title='Sentimiento', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_path}/sentiment_distribution.png', dpi=300)
        plt.close()
        
        # Heatmap de sentimientos
        plt.figure(figsize=(12, 6))
        sns.heatmap(sentiment_by_date.T, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Cantidad'})
        plt.title('Heatmap de Sentimientos Temporales', fontsize=16, fontweight='bold')
        plt.xlabel('Año', fontsize=12)
        plt.ylabel('Sentimiento', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_path}/sentiment_heatmap.png', dpi=300)
        plt.close()
        
        print(f"Visualizaciones guardadas en {output_path}")


def main_sentiment_analysis(bibtex_file='static/data/processed/merged.bib'):
    """
    Pipeline completo de análisis de sentimientos
    """
    print("="*80)
    print("ANÁLISIS DE SENTIMIENTOS Y TENDENCIAS TEMPORALES")
    print("="*80)
    
    # 1. Cargar datos del BibTeX
    try:
        import bibtexparser
        with open(bibtex_file, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
        
        entries = bib_database.entries
        df = pd.DataFrame(entries)
        print(f"\nCargados {len(df)} artículos desde {bibtex_file}")
        
        # Mostrar columnas disponibles
        print(f"📋 Columnas disponibles: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error cargando BibTeX: {e}")
        return None
    
    # 2. Seleccionar columnas de texto y fecha
    # Prioridad: abstract > title
    if 'abstract' in df.columns:
        text_column = 'abstract'
        # Contar cuántos abstracts están disponibles
        valid_abstracts = df['abstract'].notna().sum()
        print(f"📝 Usando 'abstract' como columna de texto ({valid_abstracts}/{len(df)} válidos)")
    elif 'title' in df.columns:
        text_column = 'title'
        print(f"📝 Usando 'title' como columna de texto")
    else:
        print("No se encontró columna 'abstract' ni 'title'")
        return None
    
    # Fecha: priorizar year
    if 'year' in df.columns:
        date_column = 'year'
        print(f"Usando 'year' como columna de fecha")
    else:
        date_column = None
        print("No se encontró columna 'year', análisis temporal limitado")
    
    # 3. Análisis temporal de tendencias
    print("\nAnalizando sentimientos...")
    temporal_analyzer = AnalizadorDeTendenciasTemporales()
    
    df_results = temporal_analyzer.analyze_temporal_trends(
        df, 
        text_column=text_column, 
        date_column=date_column
    )
    
    # 4. Guardar resultados
    output_dir = 'static/salidas/analizador_sentimientos'
    os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_csv(
        f'{output_dir}/resultados_sentimientos.csv',
        index=False, 
        encoding='utf-8'
    )
    print(f"\nResultados guardados en {output_dir}/resultados_sentimientos.csv")
    
    # 5. Estadísticas
    print("\nEstadísticas de Sentimiento:")
    sentiment_counts = df_results['final_label'].value_counts()
    print(sentiment_counts)
    print(f"\nTotal: {sentiment_counts.sum()} documentos")
    
    # Porcentajes
    sentiment_percentages = (sentiment_counts / sentiment_counts.sum() * 100).round(2)
    print("\nDistribución Porcentual:")
    for label, pct in sentiment_percentages.items():
        print(f"  {label}: {pct}%")
    
    print("\nConfianza Promedio por Sentimiento:")
    confidence_by_sentiment = df_results.groupby('final_label')['confidence'].mean()
    print(confidence_by_sentiment.round(4))
    
    # 6. Visualizaciones temporales
    if date_column:
        temporal_analyzer.plot_sentiment_evolution(
            df_results, 
            date_column=date_column, 
            output_path=output_dir
        )
    else:
        print("\nSin columna de fecha, omitiendo visualizaciones temporales")
    
    # 7. Topic Modeling (LDA)
    print("\nModelado de Tópicos (LDA)...")
    documents = df[text_column].fillna('').astype(str).tolist()
    
    # Filtrar documentos muy cortos
    min_length = 10
    valid_docs = [doc for doc in documents if len(doc.strip()) >= min_length]
    print(f"Documentos válidos para LDA: {len(valid_docs)}/{len(documents)}")
    
    if len(valid_docs) < 10:
        print("Pocos documentos válidos para LDA, omitiendo modelado de tópicos")
    else:
        topic_model = ModeloDeTopicos(n_topics=5)
        topic_model.fit_lda(valid_docs)
        topics = topic_model.get_topics(n_words=10)
        
        print("\nTópicos Identificados:")
        for topic in topics:
            print(f"\nTópico {topic['topic_id']}:")
            print(f"   Palabras clave: {', '.join(topic['words'][:8])}")
        
        # Guardar tópicos
        topics_df = pd.DataFrame([
            {'topic_id': t['topic_id'], 'keywords': '; '.join(t['words'][:10])}
            for t in topics
        ])
        topics_df.to_csv(f'{output_dir}/topics_lda.csv', index=False, encoding='utf-8')
        
        # Asignar tópicos a documentos válidos
        dominant_topics, _ = topic_model.predict_topics(valid_docs)
        
        # Crear columna de tópicos con -1 para documentos inválidos
        df_results['dominant_topic'] = -1
        valid_indices = [i for i, doc in enumerate(documents) if len(doc.strip()) >= min_length]
        for i, topic_id in enumerate(dominant_topics):
            if i < len(valid_indices):
                df_results.loc[valid_indices[i], 'dominant_topic'] = topic_id
        
        df_results.to_csv(f'{output_dir}/sentiment_results_with_topics.csv', index=False, encoding='utf-8')
        print(f"Resultados con tópicos guardados")
    
    # 8. Resumen final
    print("\n" + "="*80)
    print("RESUMEN DEL ANÁLISIS")
    print("="*80)
    print(f"Total de documentos analizados: {len(df_results)}")
    print(f"Positivos: {sentiment_counts.get('positive', 0)} ({sentiment_percentages.get('positive', 0)}%)")
    print(f"Neutrales: {sentiment_counts.get('neutral', 0)} ({sentiment_percentages.get('neutral', 0)}%)")
    print(f"Negativos: {sentiment_counts.get('negative', 0)} ({sentiment_percentages.get('negative', 0)}%)")
    print(f"\nResultados completos en: {output_dir}/")
    print("="*80)
    
    return df_results


if __name__ == '__main__':
    main_sentiment_analysis()
