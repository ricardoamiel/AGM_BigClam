"""
Alternativas para comparar con BigCLAM.

Particiones (un nodo, una comunidad): Louvain, Leiden, propagación de etiquetas.
Superpuestas: CPM (clique percolation), SLPA y NOCD (red neuronal de grafos
con el decodificador de BigCLAM).
"""
from collections import Counter

import networkx as nx
import numpy as np


def louvain(G, semilla=0):
    return nx.community.louvain_communities(G, seed=semilla)


def leiden(G, semilla=0):
    import igraph as ig
    import leidenalg
    g = ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()))
    particion = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=semilla)
    return [set(c) for c in particion]


def propagacion_etiquetas(G, semilla=0):
    return list(nx.community.asyn_lpa_communities(G, seed=semilla))


def cpm(G, k=3):
    """Clique percolation (Palla et al., 2005): unión de k cliques adyacentes."""
    return [set(c) for c in nx.community.k_clique_communities(G, k)]


def slpa(G, iteraciones=40, r=0.2, semilla=0):
    """
    SLPA (Xie et al., 2011). Cada nodo guarda en memoria las etiquetas que
    escucha; en cada ronda cada vecino habla una etiqueta de su memoria y el
    oyente guarda la más popular. Al final, u pertenece a toda etiqueta cuya
    frecuencia en su memoria supere r.
    """
    rng = np.random.default_rng(semilla)
    memoria = {u: [u] for u in G}
    for _ in range(iteraciones):
        for u in rng.permutation(list(G)):
            dichas = [rng.choice(memoria[v]) for v in G[u]]
            if dichas:
                memoria[u].append(Counter(dichas).most_common(1)[0][0])
    grupos = {}
    for u, etiquetas in memoria.items():
        for etiqueta, veces in Counter(etiquetas).items():
            if veces / len(etiquetas) >= r:
                grupos.setdefault(etiqueta, set()).add(u)
    unicos = {frozenset(c) for c in grupos.values() if len(c) > 1}
    return [set(c) for c in unicos if not any(c < otro for otro in unicos)]


def nocd(G, K, X=None, epocas=500, oculto=128, umbral=0.5, semilla=0, historial=False):
    """
    NOCD (Shchur y Günnemann, 2019): una GCN de dos capas produce F = GCN(A, X)
    y se entrena con la verosimilitud de Bernoulli Poisson de BigCLAM,
    balanceando aristas y no aristas. Sin atributos se usa X = A (NOCD G);
    con atributos de los nodos se usa X = atributos (NOCD X).
    """
    import torch
    torch.manual_seed(semilla)
    n = G.number_of_nodes()
    A = torch.tensor(nx.to_numpy_array(G, nodelist=range(n)), dtype=torch.float32)
    grado = (A + torch.eye(n)).sum(1)
    A_norm = (A + torch.eye(n)) / torch.sqrt(grado[:, None] * grado[None, :])
    X = A.clone() if X is None else torch.tensor(np.asarray(X), dtype=torch.float32)

    capa1 = torch.nn.Linear(X.shape[1], oculto)
    capa2 = torch.nn.Linear(oculto, K)
    normal = torch.nn.BatchNorm1d(oculto)
    parametros = [*capa1.parameters(), *capa2.parameters(), *normal.parameters()]
    opt = torch.optim.Adam(parametros, lr=1e-3)
    aristas = A.nonzero()
    no_aristas = (A + torch.eye(n) == 0).nonzero()

    def membresias():
        h = torch.relu(normal(A_norm @ capa1(X)))
        return torch.relu(A_norm @ capa2(torch.dropout(h, 0.5, normal.training)))

    def perdida_de(F):
        d_e = (F[aristas[:, 0]] * F[aristas[:, 1]]).sum(1)
        d_n = (F[no_aristas[:, 0]] * F[no_aristas[:, 1]]).sum(1)
        return -torch.log(-torch.expm1(-d_e - 1e-8)).mean() + d_n.mean()

    traza = []
    for epoca in range(epocas):
        opt.zero_grad()
        perdida = perdida_de(membresias())
        regular = perdida + 1e-2 * sum((w ** 2).sum() for w in capa1.parameters())
        regular.backward()
        opt.step()
        if historial and (epoca % 10 == 0 or epoca == epocas - 1):
            normal.eval()
            with torch.no_grad():
                F_eval = membresias()
                traza.append({"epoca": epoca, "perdida": float(perdida_de(F_eval)),
                              "F": F_eval.numpy().round(3).tolist()})
            normal.train()

    normal.eval()
    with torch.no_grad():
        F = membresias().numpy()
    detectadas = [set(np.flatnonzero(F[:, c] >= umbral)) for c in range(K)]
    return (detectadas, traza) if historial else detectadas
