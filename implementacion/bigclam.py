"""
BigCLAM: Cluster Affiliation Model for Big Networks (Yang y Leskovec, WSDM 2013).

Relaja la afiliación binaria del AGM a una matriz no negativa F (N x K):

    p(u, v) = 1 - exp(-F_u . F_v)

y la ajusta maximizando la log verosimilitud

    l(F) = sum_{(u,v) en E} log(1 - exp(-F_u . F_v))  -  sum_{(u,v) fuera de E} F_u . F_v

con ascenso por gradiente coordinado: se actualiza una fila F_u a la vez,
manteniendo fijas las demás, lo que vuelve convexo cada subproblema.
"""
import numpy as np

F_MAX = 1000.0
G_MAX = 10.0


def _log_1_menos_exp(d):
    """log(1 - exp(-d)) estable para d > 0."""
    return np.log(-np.expm1(-np.maximum(d, 1e-12)))


# ----------------------------------------------------------------------
# Subproblema de una fila: se reutiliza en CoDA (AGM dirigido)
# ----------------------------------------------------------------------
def objetivo_fila(x, Y_vecinos, suma_no_vecinos, eps_x=0.0):
    """l(F_u): verosimilitud de las aristas y no aristas que tocan a u."""
    d = Y_vecinos @ x + eps_x
    return _log_1_menos_exp(d).sum() - x @ suma_no_vecinos


def gradiente_fila(x, Y_vecinos, suma_no_vecinos, eps_x=0.0):
    """Gradiente de l(F_u), ecuación del paper."""
    d = np.maximum(Y_vecinos @ x + eps_x, 1e-12)
    return Y_vecinos.T @ (1 / np.expm1(d)) - suma_no_vecinos


def actualizar_fila(x, Y_vecinos, suma_no_vecinos, eps_x=0.0,
                    alfa=0.05, beta=0.3, intentos=15):
    """
    Un paso de gradiente proyectado con búsqueda lineal por retroceso:
    se prueba un paso t y se reduce t por beta hasta que la mejora sea
    al menos alfa * gradiente . (nuevo - x) (condición de Armijo).
    El gradiente se recorta a [-G_MAX, G_MAX]: si una arista no tiene
    comunidad compartida, su término vale 1 / (exp(eps) - 1), enorme.
    """
    g = np.clip(gradiente_fila(x, Y_vecinos, suma_no_vecinos, eps_x), -G_MAX, G_MAX)
    l0 = objetivo_fila(x, Y_vecinos, suma_no_vecinos, eps_x)
    t = 1.0
    for _ in range(intentos):
        nuevo = np.clip(x + t * g, 0, F_MAX)
        if objetivo_fila(nuevo, Y_vecinos, suma_no_vecinos, eps_x) >= l0 + alfa * g @ (nuevo - x):
            return nuevo
        t *= beta
    return x


# ----------------------------------------------------------------------
# Inicialización con vecindarios localmente mínimos (Gleich y Seshadhri)
# ----------------------------------------------------------------------
def conductancia(vecinos, conjunto, volumen_total):
    corte = sum(1 for u in conjunto for v in vecinos[u] if v not in conjunto)
    volumen = sum(len(vecinos[u]) for u in conjunto)
    return corte / max(1, min(volumen, volumen_total - volumen))


def inicializar(vecinos, K, rng=None):
    """
    N(u) = {u} ∪ vecinos(u) es localmente mínimo si su conductancia es menor
    que la de N(v) para todo vecino v. Los K vecindarios mínimos de menor
    conductancia son las semillas: F[N(u), k] = 1.
    """
    rng = np.random.default_rng(rng)
    n = len(vecinos)
    volumen_total = sum(len(v) for v in vecinos)
    phi = np.array([conductancia(vecinos, set(vecinos[u]) | {u}, volumen_total)
                    for u in range(n)])
    minimos = [u for u in range(n) if all(phi[u] <= phi[v] for v in vecinos[u])]
    minimos = sorted(minimos, key=lambda u: phi[u])[:K]
    resto = rng.permutation([u for u in range(n) if u not in minimos])
    semillas = list(minimos) + list(resto[:K - len(minimos)])

    F = np.zeros((n, K))
    for k, u in enumerate(semillas):
        F[list(vecinos[u]) + [u], k] = 1
    return F, semillas


# ----------------------------------------------------------------------
# Ajuste
# ----------------------------------------------------------------------
def log_verosimilitud(F, aristas, eps_x=0.0):
    """l(F) sobre los pares u < v; aristas es un arreglo E x 2."""
    u, v = aristas.T
    d = (F[u] * F[v]).sum(1) + eps_x
    S = F.sum(0)
    todos = (S @ S - (F * F).sum()) / 2 + eps_x * len(F) * (len(F) - 1) / 2
    return _log_1_menos_exp(d).sum() - (todos - d.sum())


def ajustar(vecinos, K, iter_max=300, tol=1e-5, eps=1e-8, rng=None, historial=False,
            ocultos=None):
    """
    Ascenso coordinado: en cada iteración se recorre cada nodo u y se
    actualiza F_u con las demás filas fijas. La suma de las filas S se guarda
    en memoria, así que sum_{v no vecino} F_v = S - F_u - sum_{v vecino} F_v
    se calcula en O(grado de u).

    ocultos[u] lista pares (u, v) reservados para validación: no cuentan ni
    como arista ni como no arista durante el ajuste.
    """
    rng = np.random.default_rng(rng)
    vecinos = [list(v) for v in vecinos]
    aristas = np.array([(u, v) for u in range(len(vecinos)) for v in vecinos[u] if u < v])
    eps_x = -np.log1p(-eps)

    F, _ = inicializar(vecinos, K, rng)
    S = F.sum(0)
    traza = [log_verosimilitud(F, aristas, eps_x)]
    fotos = [F.copy()] if historial else None

    for _ in range(iter_max):
        for u in rng.permutation(len(vecinos)):
            Y = F[vecinos[u]]
            resto = S - F[u] - Y.sum(0)
            if ocultos is not None:
                resto = resto - F[ocultos[u]].sum(0)
            nuevo = actualizar_fila(F[u], Y, resto, eps_x)
            S += nuevo - F[u]
            F[u] = nuevo
        traza.append(log_verosimilitud(F, aristas, eps_x))
        if historial:
            fotos.append(F.copy())
        if abs(traza[-1] - traza[-2]) < tol * abs(traza[-2]):
            break
    return (F, np.array(traza), fotos) if historial else (F, np.array(traza))


def umbral(vecinos):
    """delta = sqrt(-log(1 - eps)) con eps la densidad de fondo 2|E| / N(N-1)."""
    n = len(vecinos)
    eps = sum(len(v) for v in vecinos) / (n * (n - 1))
    return np.sqrt(-np.log1p(-eps))


def comunidades(F, delta):
    """El nodo u pertenece a c si F_uc >= delta."""
    return [set(np.flatnonzero(F[:, c] >= delta)) for c in range(F.shape[1])]


def elegir_K(vecinos, candidatos, fraccion=0.2, rng=0):
    """
    Criterio del paper: se reserva una fracción de los pares de nodos, se
    ajusta con el resto y se elige el K con mayor log verosimilitud sobre
    los pares reservados.
    """
    rng = np.random.default_rng(rng)
    n = len(vecinos)
    pares = np.array([(u, v) for u in range(n) for v in range(u + 1, n)])
    reservados = pares[rng.random(len(pares)) < fraccion]
    es_arista = np.array([v in set(vecinos[u]) for u, v in reservados])

    ocultos = [[] for _ in range(n)]
    for u, v in reservados:
        ocultos[u].append(v)
        ocultos[v].append(u)
    entrenamiento = [sorted(set(vecinos[u]) - set(ocultos[u])) for u in range(n)]

    puntaje = {}
    for K in candidatos:
        F, _ = ajustar(entrenamiento, K, rng=0, ocultos=ocultos)
        d = (F[reservados[:, 0]] * F[reservados[:, 1]]).sum(1) - np.log1p(-1e-8)
        puntaje[K] = np.where(es_arista, _log_1_menos_exp(d), -d).sum()
    return max(puntaje, key=puntaje.get), puntaje
