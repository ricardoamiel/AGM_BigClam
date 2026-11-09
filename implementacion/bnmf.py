"""
NMF bayesiana para comunidades (Psorakis, Roberts, Ebden y Sheldon, Phys. Rev. E 2011).

Factoriza V ≈ W H con verosimilitud de Poisson, como BigCLAM, pero pone una
prior seminormal con precisión beta_k sobre cada columna de W y fila de H.
Esa prior empuja a cero las comunidades que no explican datos (determinación
automática de relevancia, ARD): se parte de un K grande y el modelo apaga
las que sobran. Así se resuelve la debilidad de tener que fijar K.

Actualizaciones multiplicativas del paper (ecuaciones 17 a 19):

    H    <- H / (Wᵀ 1 + B H) * Wᵀ (V / WH)
    W    <- W / (1 Hᵀ + W B) * (V / WH) Hᵀ
    beta <- (N + a - 1) / (½ Σ_i w_ik² + ½ Σ_j h_kj² + b)
"""
import numpy as np


def ajustar(A, K_max=None, iteraciones=500, a=1.0, b=2.0, rng=0, historial=False):
    """A es la matriz de adyacencia; V = A + diag(grado), como en el paper."""
    rng = np.random.default_rng(rng)
    n = len(A)
    K = K_max or n
    V = A + np.diag(A.sum(1))
    W, H = rng.random((n, K)), rng.random((K, n))
    beta = np.ones(K)
    uno = np.ones((n, n))
    fotos = []
    tiny = 1e-12                      # evita 0 / 0 cuando una comunidad se apaga
    for _ in range(iteraciones):
        H = H / (W.T @ uno + beta[:, None] * H + tiny) * (W.T @ (V / (W @ H + tiny)))
        W = W / (uno @ H.T + W * beta[None, :] + tiny) * ((V / (W @ H + tiny)) @ H.T)
        beta = (n + a - 1) / (0.5 * ((W ** 2).sum(0) + (H ** 2).sum(1)) + b)
        if historial:
            fotos.append({"activas": int(activas(W, H).sum()), "peso": (W.sum(0) * H.sum(1)).tolist()})
    return (W, H, fotos) if historial else (W, H)


def activas(W, H, tol=0.02):
    """Una comunidad sigue viva si aporta una fracción no despreciable de W H."""
    peso = W.sum(0) * H.sum(1)
    return peso > tol * peso.max()


def comunidades(W, H, umbral=0.3):
    """
    Membresía suave pi_ik = w_ik / Σ_k w_ik (ecuación 20); el nodo pertenece a k
    si pi_ik supera el umbral. Con 0.3 un nodo puede estar en hasta tres.
    """
    vivas = activas(W, H)
    Wv = W[:, vivas]
    pi = Wv / np.maximum(Wv.sum(1, keepdims=True), 1e-12)
    return [set(np.flatnonzero(pi[:, k] >= umbral)) for k in range(pi.shape[1])]
