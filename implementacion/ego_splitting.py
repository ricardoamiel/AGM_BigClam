"""
Ego-splitting (Epasto, Lattanzi y Paes Leme, KDD 2017).

Convierte cualquier algoritmo de partición en uno de comunidades superpuestas:

1. Paso local: para cada nodo u se particiona su ego-net (los vecinos de u, sin u).
   Cada parte da una persona de u: una copia de u que solo ve a esa parte.
2. Cada arista (u, v) se lleva al grafo de personas uniendo la persona de u que
   contiene a v con la persona de v que contiene a u.
3. Paso global: se particiona el grafo de personas.
4. Cada comunidad de personas se traduce a sus nodos originales; como un nodo
   puede tener varias personas, puede quedar en varias comunidades.
"""
import networkx as nx


def componentes(G):
    return [set(c) for c in nx.connected_components(G)]


def louvain(G, semilla=0):
    return nx.community.louvain_communities(G, seed=semilla) if G.number_of_edges() else componentes(G)


def personas_de(G, u, local=componentes):
    """Partición de la ego-net de u; devuelve {vecino: índice de persona}."""
    ego = G.subgraph(G[u])
    return {v: i for i, parte in enumerate(local(ego)) for v in parte}


def grafo_de_personas(G, local=componentes):
    """Construye el grafo de personas; cada persona es el par (nodo, índice)."""
    persona = {u: personas_de(G, u, local) for u in G}
    P = nx.Graph()
    P.add_nodes_from((u, i) for u in G for i in set(persona[u].values()) or {0})
    P.add_edges_from(((u, persona[u][v]), (v, persona[v][u])) for u, v in G.edges())
    return P, persona


def ego_splitting(G, local=componentes, global_=louvain, minimo=3):
    P, _ = grafo_de_personas(G, local)
    comunidades = [{u for u, _ in grupo} for grupo in global_(P)]
    return [c for c in comunidades if len(c) >= minimo]
