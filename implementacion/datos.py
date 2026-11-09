"""Grafos de prueba: sintéticos con comunidades plantadas y la red ego de Facebook."""
from pathlib import Path

import networkx as nx
import numpy as np

import agm

DATOS = Path(__file__).resolve().parent.parent / "datos"


def comunidades_superpuestas(n=120, K=4, solape=0.3, p=0.3, eps=0.005, semilla=0):
    """
    Cada nodo cae en una comunidad base; una fracción solape de nodos se une
    además a una segunda comunidad. El grafo se genera con el AGM.
    """
    rng = np.random.default_rng(semilla)
    base = np.arange(n) % K
    B = np.zeros((n, K))
    B[np.arange(n), base] = 1
    for u in rng.choice(n, int(solape * n), replace=False):
        B[u, rng.choice([c for c in range(K) if c != base[u]])] = 1
    A = agm.generar(B, [p] * K, eps, rng)
    return nx.from_numpy_array(A), agm.a_conjuntos(B)


def dos_modos(n_fans=30, n_famosos=10, n_amigos=20, p=0.35, eps=0.01, semilla=0):
    """
    Grafo dirigido con una comunidad de dos modos (fans -> famosos) y una
    cohesiva (amigos <-> amigos). Algunos fans también son amigos.
    Se genera con el AGM dirigido: p(u -> v) = 1 - (1 - eps) prod (1 - p_c).
    """
    rng = np.random.default_rng(semilla)
    n = n_fans + n_famosos + n_amigos
    fans = np.arange(n_fans)
    famosos = np.arange(n_fans, n_fans + n_famosos)
    amigos = np.concatenate([np.arange(n_fans + n_famosos, n), fans[:8]])

    salida, entrada = np.zeros((n, 2)), np.zeros((n, 2))
    salida[fans, 0], entrada[famosos, 0] = 1, 1          # comunidad de dos modos
    salida[amigos, 1], entrada[amigos, 1] = 1, 1         # comunidad cohesiva

    P = 1 - (1 - eps) * np.exp(salida @ np.diag(np.log1p(-np.array([p, p]))) @ entrada.T)
    np.fill_diagonal(P, 0)
    A = rng.random((n, n)) < P
    G = nx.from_numpy_array(A.astype(int), create_using=nx.DiGraph)
    verdad = {"dos modos": (set(fans), set(famosos)), "cohesiva": (set(amigos), set(amigos))}
    return G, verdad


def facebook_ego(ego=414, atributos=False):
    """
    Red ego de Facebook de SNAP (McAuley y Leskovec, 2012) con sus círculos
    sociales como verdad de referencia. Se reindexan los nodos desde 0.
    Con atributos=True devuelve además la matriz binaria de rasgos del perfil.
    """
    carpeta = DATOS / "facebook" / "facebook"
    G = nx.read_edgelist(carpeta / f"{ego}.edges", nodetype=int)
    circulos = []
    for linea in (carpeta / f"{ego}.circles").read_text().splitlines():
        circulos.append({int(x) for x in linea.split()[1:]})
    G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
    indice = {u: i for i, u in enumerate(sorted(G))}
    G = nx.relabel_nodes(G, indice)
    circulos = [{indice[u] for u in c if u in indice} for c in circulos]
    circulos = [c for c in circulos if len(c) >= 3]
    if not atributos:
        return G, circulos
    rasgos = np.loadtxt(carpeta / f"{ego}.feat", dtype=int)
    X = np.zeros((len(indice), rasgos.shape[1] - 1))
    for fila in rasgos:
        if fila[0] in indice:
            X[indice[fila[0]]] = fila[1:]
    return G, circulos, X


def vecinos(G):
    """Lista de adyacencia indexada por nodo, la entrada que espera bigclam."""
    return [sorted(G[u]) for u in range(G.number_of_nodes())]
