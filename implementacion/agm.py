"""
AGM: Community-Affiliation Graph Model (Yang y Leskovec, ICDM 2012).

El modelo parte de un grafo bipartito de afiliación B (nodos por comunidades)
y de una probabilidad p_c por comunidad. Dos nodos u, v se conectan con

    p(u, v) = 1 - (1 - eps) * prod_{c en C_uv} (1 - p_c)

donde C_uv son las comunidades que comparten y eps es la comunidad epsilon,
que permite aristas entre nodos sin comunidades en común.

En todo el módulo B es una matriz binaria de N x K y A la matriz de adyacencia.
"""
import numpy as np
from scipy.optimize import minimize

from bigclam import inicializar


def a_matriz(membresias, n):
    """Convierte una lista de conjuntos de nodos en la matriz binaria B (N x K)."""
    B = np.zeros((n, len(membresias)))
    for c, nodos in enumerate(membresias):
        B[list(nodos), c] = 1
    return B


def a_conjuntos(B):
    """Inversa de a_matriz: una lista con los nodos de cada columna de B."""
    return [set(np.flatnonzero(B[:, c])) for c in range(B.shape[1])]


def probabilidades(B, p, eps):
    """Matriz N x N con p(u, v) para cada par de nodos."""
    log_no_arista = B @ np.diag(np.log1p(-np.asarray(p))) @ B.T
    P = 1 - (1 - eps) * np.exp(log_no_arista)
    np.fill_diagonal(P, 0)
    return P


def generar(B, p, eps=0.0, rng=None):
    """Proceso generativo: lanza una moneda por cada par u < v."""
    rng = np.random.default_rng(rng)
    P = probabilidades(B, p, eps)
    A = np.triu(rng.random(P.shape) < P, k=1)
    return (A | A.T).astype(float)


# ----------------------------------------------------------------------
# Máxima verosimilitud
# ----------------------------------------------------------------------
# Con el cambio de variable 1 - p_c = exp(-x_c), la suma del exponente es
# lineal: s_uv = sum_c B_uc B_vc x_c + x_eps, y la log verosimilitud queda
#
#     l = sum_{(u,v) en E} log(1 - exp(-s_uv))  -  sum_{(u,v) fuera de E} s_uv
#
# que es cóncava en x para un B fijo.

def _exponente(B, x, eps):
    S = B @ np.diag(x) @ B.T - np.log1p(-eps)
    return np.maximum(S, 1e-12)


def log_verosimilitud(A, B, p, eps):
    """log P(G | B, p), sumada sobre los pares u < v."""
    x = -np.log1p(-np.clip(p, 0, 1 - 1e-12))
    S = _exponente(B, x, eps)
    L = np.where(A > 0, np.log(-np.expm1(-S)), -S)
    return np.triu(L, k=1).sum()


def ajustar_p(A, B, eps):
    """Paso 1 del ajuste: con B fijo, el p_c que maximiza la verosimilitud."""
    mascara = np.triu(np.ones_like(A), k=1)

    def costo(x):
        S = _exponente(B, x, eps)
        L = np.where(A > 0, np.log(-np.expm1(-S)), -S)
        W = np.where(A > 0, 1 / np.expm1(S), -1.0) * mascara
        gradiente = np.einsum("uk,uv,vk->k", B, W, B)
        return -(L * mascara).sum(), -gradiente

    x0 = np.full(B.shape[1], 0.5)
    x = minimize(costo, x0, jac=True, method="L-BFGS-B",
                 bounds=[(0, 20)] * B.shape[1]).x
    return 1 - np.exp(-x)


def proponer(B, rng):
    """Paso 2 del ajuste: una transición LEAVE, JOIN o SWITCH sobre B."""
    B = B.copy()
    u = rng.integers(B.shape[0])
    dentro, fuera = np.flatnonzero(B[u]), np.flatnonzero(B[u] == 0)
    movimiento = rng.choice(["LEAVE", "JOIN", "SWITCH"])
    if movimiento == "LEAVE" and len(dentro):
        B[u, rng.choice(dentro)] = 0
    elif movimiento == "JOIN" and len(fuera):
        B[u, rng.choice(fuera)] = 1
    elif movimiento == "SWITCH" and len(dentro) and len(fuera):
        B[u, rng.choice(dentro)], B[u, rng.choice(fuera)] = 0, 1
    return B, movimiento


def ajustar(A, K, pasos=20000, cada=500, rng=None):
    """
    Ajuste del AGM por ascenso coordinado, como en el paper:
    alterna optimizar p con B fijo y explorar B con Metropolis Hastings.
    B parte de vecindarios localmente mínimos, las mismas semillas de BigCLAM.
    Devuelve el mejor B encontrado, sus p y la traza de la log verosimilitud.
    """
    rng = np.random.default_rng(rng)
    n = A.shape[0]
    eps = A.sum() / (n * (n - 1))
    B, _ = inicializar([np.flatnonzero(fila) for fila in A], K, rng)
    p = ajustar_p(A, B, eps)
    l = log_verosimilitud(A, B, p, eps)
    mejor, traza = (l, B, p), [l]

    for paso in range(1, pasos + 1):
        B_nuevo, _ = proponer(B, rng)
        l_nuevo = log_verosimilitud(A, B_nuevo, p, eps)
        if np.log(rng.random()) < l_nuevo - l:          # aceptación de MH
            B, l = B_nuevo, l_nuevo
        if paso % cada == 0:
            p = ajustar_p(A, B, eps)
            l = log_verosimilitud(A, B, p, eps)
        if l > mejor[0]:
            mejor = (l, B, p)
        traza.append(l)
    return mejor[1], mejor[2], np.array(traza)
