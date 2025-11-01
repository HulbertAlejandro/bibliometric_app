"""
Análisis de Grafos con Algoritmos Voraces y Teoría de Grafos
Implementa: Kruskal, Prim, Dijkstra, PageRank, Louvain
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from collections import defaultdict, Counter
import json

# Community detection
try:
    import community.community_louvain as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("Warning: python-louvain not installed. Use: pip install python-louvain")

# Interactive visualization
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed for interactive graphs")


class GraphBuilder:
    """
    Constructor de grafos de co-autoría y co-citación
    """
    
    def __init__(self):
        self.G = nx.Graph()
        self.authors_dict = defaultdict(list)
        self.citations_dict = defaultdict(set)
    
    def construir_coautor_grafo(self, df, author_column='author'):
        """
        Construye grafo de co-autoría
        Complejidad: O(n·k²) donde n=papers, k=autores por paper
        """
        self.G = nx.Graph()
        
        for idx, row in df.iterrows():
            authors = self._parse_authors(row.get(author_column, ''))
            paper_id = row.get('ID', f'paper_{idx}')
            
            # Agregar nodos (autores)
            for author in authors:
                if not self.G.has_node(author):
                    # ✅ CAMBIO: Convertir lista a string separado por punto y coma
                    self.G.add_node(author, papers=paper_id, type='author')
                else:
                    # ✅ CAMBIO: Concatenar IDs con punto y coma
                    existing_papers = self.G.nodes[author].get('papers', '')
                    self.G.nodes[author]['papers'] = f"{existing_papers}; {paper_id}"
            
            # Agregar aristas entre co-autores
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    if self.G.has_edge(author1, author2):
                        self.G[author1][author2]['weight'] += 1
                    else:
                        self.G.add_edge(author1, author2, weight=1)
        
        print(f"✅ Grafo construido: {self.G.number_of_nodes()} nodos, {self.G.number_of_edges()} aristas")
        return self.G
    
    def _parse_authors(self, author_string):
        """
        Parsea string de autores
        """
        if pd.isna(author_string):
            return []
        
        # Separar por 'and' o comas
        authors = str(author_string).replace(' and ', ',').split(',')
        authors = [a.strip() for a in authors if a.strip()]
        return authors
    
    def construir_grafo_citas(self, df, citation_column='references'):
        """
        Construye grafo de citaciones
        Complejidad: O(n·c) donde n=papers, c=citas por paper
        """
        self.G = nx.DiGraph()  # Grafo dirigido para citaciones
        
        for idx, row in df.iterrows():
            paper_id = row.get('ID', f'paper_{idx}')
            title = row.get('title', paper_id)
            
            self.G.add_node(paper_id, title=title, type='paper')
            
            # Agregar aristas de citación
            citations = self._parse_citations(row.get(citation_column, ''))
            for cited_paper in citations:
                if not self.G.has_node(cited_paper):
                    self.G.add_node(cited_paper, title=cited_paper, type='paper')
                
                self.G.add_edge(paper_id, cited_paper, type='citation')
        
        return self.G
    
    def _parse_citations(self, citation_string):
        """
        Parsea referencias citadas
        """
        if pd.isna(citation_string):
            return []
        
        citations = str(citation_string).split(';')
        citations = [c.strip() for c in citations if c.strip()]
        return citations


class GreedyAlgorithms:
    """
    Implementación de algoritmos voraces para grafos
    """
    
    @staticmethod
    def kruskal_mst(G):
        """
        Algoritmo de Kruskal para Minimum Spanning Tree
        Complejidad: O(E log E) donde E = número de aristas
        
        Algoritmo Voraz:
        1. Ordenar aristas por peso creciente
        2. Para cada arista, agregar si no forma ciclo
        """
        if G.is_directed():
            G = G.to_undirected()
        
        mst = nx.minimum_spanning_tree(G, algorithm='kruskal')
        
        total_weight = sum(data.get('weight', 1) for _, _, data in mst.edges(data=True))
        
        print(f"\n🌳 Kruskal MST:")
        print(f"  Aristas originales: {G.number_of_edges()}")
        print(f"  Aristas en MST: {mst.number_of_edges()}")
        print(f"  Peso total: {total_weight}")
        
        return mst
    
    @staticmethod
    def prim_mst(G):
        """
        Algoritmo de Prim para MST
        Complejidad: O(E log V) con heap binario
        
        Algoritmo Voraz:
        1. Comenzar con nodo arbitrario
        2. Agregar arista mínima que conecta árbol con nuevo nodo
        """
        if G.is_directed():
            G = G.to_undirected()
        
        mst = nx.minimum_spanning_tree(G, algorithm='prim')
        
        total_weight = sum(data.get('weight', 1) for _, _, data in mst.edges(data=True))
        
        print(f"\n🌳 Prim MST:")
        print(f"  Aristas en MST: {mst.number_of_edges()}")
        print(f"  Peso total: {total_weight}")
        
        return mst
    
    @staticmethod
    def dijkstra_shortest_paths(G, source):
        """
        Algoritmo de Dijkstra para caminos más cortos
        Complejidad: O((V + E) log V) con heap de Fibonacci
        
        Algoritmo Voraz:
        1. Mantener distancias tentativas a todos los nodos
        2. Seleccionar nodo con menor distancia no visitado
        3. Actualizar distancias de vecinos
        """
        try:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
            paths = nx.single_source_dijkstra_path(G, source, weight='weight')
            
            print(f"\n🛤️  Dijkstra desde '{source}':")
            print(f"  Nodos alcanzables: {len(lengths)}")
            
            # Mostrar 5 caminos más cortos
            sorted_lengths = sorted(lengths.items(), key=lambda x: x[1])[:5]
            for node, dist in sorted_lengths:
                if node != source:
                    print(f"  → {node}: distancia {dist}")
            
            return lengths, paths
        except nx.NodeNotFound:
            print(f"❌ Nodo '{source}' no encontrado en el grafo")
            return {}, {}
    
    @staticmethod
    def greedy_coloring(G):
        """
        Coloración voraz de grafos
        Complejidad: O(V + E)
        
        Algoritmo Voraz:
        1. Ordenar nodos por grado decreciente
        2. Asignar color más bajo disponible a cada nodo
        """
        coloring = nx.greedy_color(G, strategy='largest_first')
        
        num_colors = max(coloring.values()) + 1
        
        print(f"\n🎨 Coloración Voraz:")
        print(f"  Número de colores usados: {num_colors}")
        print(f"  Número cromático estimado: {num_colors}")
        
        # Distribución de colores
        color_counts = Counter(coloring.values())
        print(f"  Distribución: {dict(color_counts)}")
        
        return coloring


class GraphMetrics:
    """
    Métricas avanzadas de grafos
    """
    
    @staticmethod
    def pagerank_analysis(G, alpha=0.85, max_iter=100):
        """
        PageRank para identificar nodos influyentes
        Complejidad: O(k·(V + E)) donde k=iteraciones
        """
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, weight='weight')
        
        print(f"\n⭐ PageRank Analysis:")
        top_10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("  Top 10 nodos más influyentes:")
        for i, (node, score) in enumerate(top_10, 1):
            print(f"  {i}. {node}: {score:.4f}")
        
        return pagerank
    
    @staticmethod
    def betweenness_centrality(G):
        """
        Centralidad de intermediación - identifica "puentes"
        Complejidad: O(V·E) para grafos no ponderados
        """
        betweenness = nx.betweenness_centrality(G, weight='weight')
        
        print(f"\n🌉 Betweenness Centrality (Papers Puente):")
        top_10 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (node, score) in enumerate(top_10, 1):
            print(f"  {i}. {node}: {score:.4f}")
        
        return betweenness


class GraphVisualizer:
    """
    Visualización de grafos
    """
    
    @staticmethod
    def plot_graph(G, output_path, title="Graph", layout='spring', node_color_attr=None, figsize=(16, 12)):
        """
        Visualiza grafo con NetworkX + Matplotlib
        """
        plt.figure(figsize=figsize)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Colores de nodos
        if node_color_attr and node_color_attr in nx.get_node_attributes(G, node_color_attr):
            node_colors = [G.nodes[node].get(node_color_attr, 0) for node in G.nodes()]
            node_colors = plt.cm.viridis(np.array(node_colors) / max(node_colors))
        else:
            node_colors = '#1f77b4'
        
        # Tamaño de nodos basado en grado
        node_sizes = [G.degree(node) * 50 + 100 for node in G.nodes()]
        
        # Dibujar
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray')
        
        # Labels solo para nodos importantes
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20]
        labels = {node: node for node, _ in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Grafo guardado: {output_path}")
    
    @staticmethod
    def plot_interactive(G, output_path, title="Interactive Graph"):
        """
        Visualización interactiva con Plotly
        """
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly no disponible")
            return
        
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Nodes
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Grado: {G.degree(node)}")
            node_sizes.append(G.degree(node) * 10 + 10)
        
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=node_sizes,
                    colorbar=dict(
                        thickness=15,
                        title='Conexiones',
                        xanchor='left'
                    ),
                    line_width=2
                )
            )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        fig.write_html(output_path)
        print(f"✅ Grafo interactivo guardado: {output_path}")


def main_analizador_grafos(bibtex_file='static/data/processed/merged.bib'):
    """
    Pipeline completo de análisis de grafos
    """
    print("="*80)
    print("ANÁLISIS DE GRAFOS CON ALGORITMOS VORACES")
    print("="*80)
    
    # 1. Cargar datos
    try:
        import bibtexparser
        with open(bibtex_file, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
        
        entries = bib_database.entries
        df = pd.DataFrame(entries)
        print(f"\n✅ Cargados {len(df)} artículos")
    except Exception as e:
        print(f"❌ Error: {e}")
        # Datos de ejemplo
        df = pd.DataFrame({
            'ID': ['paper1', 'paper2', 'paper3'],
            'author': ['Smith, J. and Doe, A.', 'Doe, A. and Johnson, B.', 'Smith, J. and Lee, C.'],
            'title': ['AI in Education', 'Machine Learning', 'Deep Learning'],
            'year': [2022, 2023, 2024]
        })
    
    output_dir = 'static/salidas/grafo_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Construir grafo de co-autoría
    print("\n🔨 Construyendo grafo de co-autoría...")
    builder = GraphBuilder()
    K = 10
    author_counts = (
        df['author']
        .dropna()
        .apply(lambda x: str(x).split(' and ')[0].strip() if isinstance(x, str) else "")
        .value_counts()
    )
    top_authors = set(author_counts.head(K).index)
    df_top = df[df['author'].apply(lambda x: str(x).split(' and ')[0].strip() if isinstance(x, str) else None).isin(top_authors)]
    G = builder.construir_coautor_grafo(df_top, author_column='author')
    
    # Guardar grafo
    nx.write_gexf(G, f'{output_dir}/coauthorship_graph.gexf')
    
    # 3. Algoritmos Voraces
    greedy = GreedyAlgorithms()
    
    # Kruskal MST
    mst_kruskal = greedy.kruskal_mst(G)
    
    # Prim MST
    mst_prim = greedy.prim_mst(G)
    
    # Dijkstra
    if G.number_of_nodes() > 0:
        source_node = list(G.nodes())[0]
        lengths, paths = greedy.dijkstra_shortest_paths(G, source_node)
    
    # Coloración
    coloring = greedy.greedy_coloring(G)
    
    # 4. Métricas
    metrics = GraphMetrics()
    
    pagerank = metrics.pagerank_analysis(G)
    betweenness = metrics.betweenness_centrality(G)
    
    # 5. Guardar resultados
    results_df = pd.DataFrame({
        'author': list(G.nodes()),
        'degree': [G.degree(n) for n in G.nodes()],
        'pagerank': [pagerank.get(n, 0) for n in G.nodes()],
        'betweenness': [betweenness.get(n, 0) for n in G.nodes()],
        'color': [coloring.get(n, -1) for n in G.nodes()]
    })
    
    results_df.to_csv(f'{output_dir}/graph_metrics.csv', index=False)
    print(f"\n✅ Métricas guardadas: {output_dir}/graph_metrics.csv")
    
    # 6. Visualizaciones
    visualizer = GraphVisualizer()
    
    print("\n🎨 Generando visualizaciones...")
    
    # Grafo completo
    visualizer.plot_graph(G, f'{output_dir}/coauthorship_network.png', 
                         title='Red de Co-autoría', layout='spring')
    
    # MST de Kruskal
    visualizer.plot_graph(mst_kruskal, f'{output_dir}/mst_kruskal.png',
                         title='Minimum Spanning Tree (Kruskal)', layout='kamada_kawai')
    
    # Grafo interactivo
    visualizer.plot_interactive(G, f'{output_dir}/interactive_network.html',
                               title='Red de Co-autoría Interactiva')
    
    print(f"\n✅ Análisis completado. Resultados en {output_dir}")
    
    return G, results_df


if __name__ == '__main__':
    main_analizador_grafos()
