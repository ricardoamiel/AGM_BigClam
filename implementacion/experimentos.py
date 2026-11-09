"""
Corre todos los experimentos y guarda los resultados en resultados/*.json.
Las figuras de la presentación y el visualizador leen esos archivos.

Uso:  python experimentos.py
"""
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

import agm
import alternativas as alt
import bigclam
import bnmf
import coda
import ego_splitting
import datos
import metricas

SALIDA = Path(__file__).resolve().parent.parent / "resultados"
SALIDA.mkdir(exist_ok=True)


def guardar(nombre, objeto):
    def limpiar(x):
        if isinstance(x, (set, frozenset)):
            return sorted(int(v) if isinstance(v, (int, np.integer)) else v for v in x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    (SALIDA / f"{nombre}.json").write_text(json.dumps(objeto, default=limpiar, indent=1))
    print("guardado", nombre)


def densidad_por_comunidades_compartidas():
    """Probabilidad empírica de arista según cuántas comunidades comparte el par."""
    G, verdad = datos.comunidades_superpuestas(n=600, K=6, solape=0.6, p=0.08, eps=0.002)
    B = agm.a_matriz(verdad, 600)
    compartidas = B @ B.T
    A = nx.to_numpy_array(G, nodelist=range(600))
    iu = np.triu_indices(600, 1)
    k, a = compartidas[iu], A[iu]
    return {"k": [0, 1, 2], "p": [a[k == j].mean() for j in (0, 1, 2)],
            "teorico": [0.002, 1 - 0.998 * 0.92, 1 - 0.998 * 0.92 ** 2]}


def ejemplo_bigclam():
    """Grafo chico con historial completo de F para las figuras paso a paso."""
    G, verdad = datos.comunidades_superpuestas(n=30, K=3, solape=0.3, p=0.5, eps=0.02, semilla=3)
    V = datos.vecinos(G)
    F, traza, fotos = bigclam.ajustar(V, 3, rng=0, historial=True)
    delta = bigclam.umbral(V)
    pos = nx.spring_layout(G, seed=4)
    return {"n": 30, "aristas": list(G.edges()), "verdad": verdad, "traza": traza,
            "fotos": fotos, "delta": delta, "pos": [pos[u].tolist() for u in range(30)],
            "detectadas": bigclam.comunidades(F, delta),
            "f1": metricas.f1_promedio(verdad, bigclam.comunidades(F, delta))}


def ajuste_agm_contra_bigclam():
    G, verdad = datos.comunidades_superpuestas(n=45, K=3, solape=0.25, p=0.5, eps=0.01, semilla=1)
    A = nx.to_numpy_array(G, nodelist=range(45))
    t = time.time()
    B, p, traza = agm.ajustar(A, 3, pasos=20000, cada=50, rng=0)
    t_agm = time.time() - t
    V = datos.vecinos(G)
    t = time.time()
    F, traza_b = bigclam.ajustar(V, 3, rng=0)
    t_big = time.time() - t
    return {"traza_agm": traza[::50], "p_agm": p, "f1_agm": metricas.f1_promedio(verdad, agm.a_conjuntos(B)),
            "t_agm": t_agm, "traza_bigclam": traza_b, "t_bigclam": t_big,
            "f1_bigclam": metricas.f1_promedio(verdad, bigclam.comunidades(F, bigclam.umbral(V)))}


def karate():
    G = nx.karate_club_graph()
    V = datos.vecinos(G)
    F, traza = bigclam.ajustar(V, 2, rng=0)
    club = [{u for u in G if G.nodes[u]["club"] == c} for c in ("Mr. Hi", "Officer")]
    pos = nx.spring_layout(G, seed=7)
    return {"aristas": list(G.edges()), "F": F, "delta": bigclam.umbral(V), "club": club,
            "pos": [pos[u].tolist() for u in G]}


def elegir_k():
    G, verdad = datos.comunidades_superpuestas()
    K, puntaje = bigclam.elegir_K(datos.vecinos(G), range(1, 9))
    return {"K": K, "K_real": len(verdad), "puntaje": puntaje}


def detectar_bigclam(G, K):
    V = datos.vecinos(G)
    F, _ = bigclam.ajustar(V, K, rng=0)
    return bigclam.comunidades(F, bigclam.umbral(V))


METODOS = {
    "BigCLAM": detectar_bigclam,
    "Louvain": lambda G, K: alt.louvain(G),
    "Leiden": lambda G, K: alt.leiden(G),
    "Propagación de etiquetas": lambda G, K: alt.propagacion_etiquetas(G),
    "CPM (k = 3)": lambda G, K: alt.cpm(G, 3),
    "SLPA": lambda G, K: alt.slpa(G),
    "NOCD": lambda G, K: alt.nocd(G, K),
    "Ego-splitting": lambda G, K: ego_splitting.ego_splitting(G),
    "NMF bayesiana": lambda G, K: detectar_bnmf(G),
}


def detectar_bnmf(G):
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    W, H = bnmf.ajustar(A, K_max=30)
    return bnmf.comunidades(W, H)


def benchmark_solapamiento(semillas=(0, 1, 2)):
    niveles = [0.0, 0.2, 0.4, 0.6, 0.8]
    f1 = {m: [] for m in METODOS}
    for s in niveles:
        grafos = [datos.comunidades_superpuestas(solape=s, semilla=r) for r in semillas]
        for nombre, metodo in METODOS.items():
            f1[nombre].append(np.mean([metricas.f1_promedio(v, metodo(G, 4)) for G, v in grafos]))
        print("solape", s, {k: round(v[-1], 3) for k, v in f1.items()})
    return {"solape": niveles, "f1": f1}


def facebook(egos=(414, 686, 348, 0)):
    tabla = {}
    for ego in egos:
        G, circulos = datos.facebook_ego(ego)
        fila = {"nodos": G.number_of_nodes(), "aristas": G.number_of_edges(), "circulos": len(circulos)}
        for nombre, metodo in METODOS.items():
            t = time.time()
            fila[nombre] = metricas.f1_promedio(circulos, metodo(G, len(circulos)))
            fila["t " + nombre] = time.time() - t
        _, _, X = datos.facebook_ego(ego, atributos=True)
        fila["NOCD con atributos"] = metricas.f1_promedio(circulos, alt.nocd(G, len(circulos), X=X))
        tabla[ego] = fila
        print("ego", ego, {k: round(v, 3) for k, v in fila.items() if not k.startswith("t ")})
    return tabla


def dirigido():
    D, verdad = datos.dos_modos()
    n = D.number_of_nodes()
    salientes = [sorted(D.successors(u)) for u in range(n)]
    entrantes = [sorted(D.predecessors(u)) for u in range(n)]
    F, H, traza = coda.ajustar(salientes, entrantes, 2, rng=0)
    delta = np.sqrt(-np.log1p(-D.number_of_edges() / (n * (n - 1))))
    # BigCLAM sobre el grafo sin dirección, para contrastar
    V = datos.vecinos(D.to_undirected())
    Fb, _ = bigclam.ajustar(V, 2, rng=0)
    return {"n": n, "aristas": list(D.edges()), "F": F, "H": H, "traza": traza, "delta": delta,
            "verdad": verdad, "coda": coda.comunidades(F, H, delta),
            "bigclam": bigclam.comunidades(Fb, bigclam.umbral(V))}


def k_automatico():
    """NMF bayesiana: cuántas comunidades deja vivas según el solape, partiendo de 30."""
    filas = []
    for s in [0.0, 0.2, 0.4, 0.6, 0.8]:
        G, verdad = datos.comunidades_superpuestas(solape=s)
        A = nx.to_numpy_array(G, nodelist=range(120))
        W, H = bnmf.ajustar(A, K_max=30)
        filas.append({"solape": s, "K": int(bnmf.activas(W, H).sum()), "K_real": len(verdad)})
    G, _ = datos.comunidades_superpuestas(solape=0.0)
    _, _, fotos = bnmf.ajustar(nx.to_numpy_array(G, nodelist=range(120)), K_max=30, iteraciones=200,
                               historial=True)
    return {"por_solape": filas, "activas": [f["activas"] for f in fotos]}


def ego_ejemplo():
    """Grafo de la figura 1 del paper de Ego-splitting, con letras como nodos."""
    G = nx.Graph([("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("c", "e"), ("c", "f"), ("d", "e"),
                  ("d", "f"), ("e", "f"), ("f", "g"), ("f", "h"), ("g", "h")])
    P, persona = ego_splitting.grafo_de_personas(G)
    return {"aristas": list(G.edges()), "personas": {u: persona[u] for u in G},
            "aristas_personas": [[f"{a}{i + 1}", f"{b}{j + 1}"] for (a, i), (b, j) in P.edges()],
            "comunidades": ego_splitting.ego_splitting(G, global_=ego_splitting.componentes, minimo=2)}


def nocd_visualizador():
    """Entrena NOCD guardando F cada 10 épocas y lo escribe como datos del visualizador."""
    G, verdad = datos.comunidades_superpuestas(n=24, K=3, solape=0.3, p=0.55, eps=0.02, semilla=3)
    _, traza = alt.nocd(G, 3, epocas=300, historial=True)
    pos = nx.spring_layout(G, seed=4)
    salida = {"n": 24, "aristas": list(G.edges()), "verdad": verdad, "traza": traza,
              "pos": [pos[u].round(4).tolist() for u in range(24)]}
    destino = SALIDA.parent / "visualizador" / "datos_nocd.js"
    destino.write_text("// generado por implementacion/experimentos.py (tarea nocd_visualizador)\n"
                       "const DATOS_NOCD = " + json.dumps(salida, default=lambda x: sorted(int(v) for v in x)) + ";\n")
    return {"epocas": [t["epoca"] for t in traza], "perdida": [t["perdida"] for t in traza]}


if __name__ == "__main__":
    import sys
    solo = sys.argv[1:]
    tareas = {"densidad_solapamiento": densidad_por_comunidades_compartidas,
              "ejemplo_bigclam": ejemplo_bigclam, "agm_contra_bigclam": ajuste_agm_contra_bigclam,
              "karate": karate, "elegir_k": elegir_k, "dirigido": dirigido,
              "benchmark": benchmark_solapamiento, "facebook": facebook, "k_automatico": k_automatico,
              "ego_ejemplo": ego_ejemplo, "nocd_visualizador": nocd_visualizador}
    for nombre, tarea in tareas.items():
        if not solo or nombre in solo:
            guardar(nombre, tarea())
