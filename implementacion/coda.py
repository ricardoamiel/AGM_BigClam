"""
AGM dirigido y CoDA (Yang, McAuley y Leskovec, WSDM 2014).

Cada nodo tiene dos vectores de membresía no negativos:
    F_u  fuerza con la que u envía aristas a la comunidad (membresía saliente)
    H_v  fuerza con la que v recibe aristas de la comunidad (membresía entrante)

    p(u -> v) = 1 - exp(-F_u . H_v)

Una comunidad cohesiva tiene F y H altos en los mismos nodos; una comunidad
de dos modos (fans que siguen a celebridades) tiene F alto en unos nodos y
H alto en otros. El ajuste reutiliza el subproblema de fila de BigCLAM.
"""
import numpy as np

from bigclam import actualizar_fila, inicializar, _log_1_menos_exp


def log_verosimilitud(F, H, aristas, eps_x=0.0):
    """l(F, H) sobre todos los pares ordenados u != v."""
    u, v = aristas.T
    d = (F[u] * H[v]).sum(1) + eps_x
    n = len(F)
    todos = F.sum(0) @ H.sum(0) - (F * H).sum() + eps_x * n * (n - 1)
    return _log_1_menos_exp(d).sum() - (todos - d.sum())


def ajustar(salientes, entrantes, K, iter_max=300, tol=1e-5, eps=1e-8, rng=None):
    """
    salientes[u] lista los v con u -> v; entrantes[v] lista los u con u -> v.
    En cada iteración se actualiza F_u con H fijo y luego H_v con F fijo.
    """
    rng = np.random.default_rng(rng)
    n = len(salientes)
    eps_x = -np.log1p(-eps)
    aristas = np.array([(u, v) for u in range(n) for v in salientes[u]])

    # semillas sobre la versión no dirigida, como propone el paper
    no_dirigido = [sorted(set(salientes[u]) | set(entrantes[u])) for u in range(n)]
    semilla, _ = inicializar(no_dirigido, K, rng)
    F = semilla * np.array([len(s) > 0 for s in salientes])[:, None]
    H = semilla * np.array([len(e) > 0 for e in entrantes])[:, None]

    traza = [log_verosimilitud(F, H, aristas, eps_x)]
    for _ in range(iter_max):
        SH = H.sum(0)
        for u in rng.permutation(n):
            Y = H[salientes[u]]
            F[u] = actualizar_fila(F[u], Y, SH - H[u] - Y.sum(0), eps_x)
        SF = F.sum(0)
        for v in rng.permutation(n):
            Y = F[entrantes[v]]
            H[v] = actualizar_fila(H[v], Y, SF - F[v] - Y.sum(0), eps_x)
        traza.append(log_verosimilitud(F, H, aristas, eps_x))
        if abs(traza[-1] - traza[-2]) < tol * abs(traza[-2]):
            break
    return F, H, np.array(traza)


def comunidades(F, H, delta):
    """Para cada comunidad, quién envía (F >= delta) y quién recibe (H >= delta)."""
    return [(set(np.flatnonzero(F[:, c] >= delta)), set(np.flatnonzero(H[:, c] >= delta)))
            for c in range(F.shape[1])]
