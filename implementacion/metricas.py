"""Métricas para comparar comunidades superpuestas con la verdad de referencia."""
import numpy as np


def f1(a, b):
    a, b = set(a), set(b)
    comunes = len(a & b)
    return 0.0 if comunes == 0 else 2 * comunes / (len(a) + len(b))


def f1_promedio(verdad, detectadas):
    """
    F1 promedio de Yang y Leskovec (2013): cada comunidad real se empareja con
    la detectada más parecida y viceversa, y se promedian ambos sentidos.
    """
    verdad = [c for c in verdad if c]
    detectadas = [c for c in detectadas if c]
    if not verdad or not detectadas:
        return 0.0
    M = np.array([[f1(a, b) for b in detectadas] for a in verdad])
    return (M.max(1).mean() + M.max(0).mean()) / 2


def nodos_en_solapamiento(comunidades):
    """Nodos que pertenecen a dos o más comunidades."""
    conteo = {}
    for c in comunidades:
        for u in c:
            conteo[u] = conteo.get(u, 0) + 1
    return {u for u, k in conteo.items() if k >= 2}
