// Modelos AGM, BigCLAM y CoDA en JavaScript, equivalentes a implementacion/*.py.
"use strict";

// ------------------------------------------------------------------
// utilidades
// ------------------------------------------------------------------
function generador(semilla) {
  // mulberry32: números pseudoaleatorios reproducibles
  let a = semilla >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function permutar(n, rng) {
  const p = d3.range(n);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

const punto = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const sumaVec = (a, b) => a.map((x, i) => x + b[i]);
const restaVec = (a, b) => a.map((x, i) => x - b[i]);
const escalarVec = (a, k) => a.map((x) => x * k);
const ceros = (k) => new Array(k).fill(0);
const log1menosExp = (d) => Math.log(-Math.expm1(-Math.max(d, 1e-12)));

// ------------------------------------------------------------------
// AGM
// ------------------------------------------------------------------
const AGM = {
  // membresias: lista de arreglos de nodos; p: probabilidad por comunidad
  probabilidad(u, v, membresias, p, eps) {
    let q = 1 - eps;
    const compartidas = [];
    membresias.forEach((nodos, c) => {
      if (nodos.includes(u) && nodos.includes(v)) {
        q *= 1 - p[c];
        compartidas.push(c);
      }
    });
    return { p: 1 - q, compartidas };
  },

  pares(n) {
    const pares = [];
    for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) pares.push([u, v]);
    return pares;
  },

  logVerosimilitud(n, aristas, membresias, p, eps) {
    const E = new Set(aristas.map(([u, v]) => u + "," + v));
    let l = 0;
    for (const [u, v] of AGM.pares(n)) {
      const q = Math.min(Math.max(AGM.probabilidad(u, v, membresias, p, eps).p, 1e-12), 1 - 1e-12);
      l += E.has(u + "," + v) ? Math.log(q) : Math.log(1 - q);
    }
    return l;
  },

  // grafo con comunidades plantadas: cada nodo en una base y una fracción en dos
  plantado(n, K, solape, p, eps, semilla) {
    const rng = generador(semilla);
    const membresias = d3.range(K).map(() => []);
    for (let u = 0; u < n; u++) membresias[u % K].push(u);
    const extra = permutar(n, rng).slice(0, Math.round(solape * n));
    for (const u of extra) {
      const opciones = d3.range(K).filter((c) => c !== u % K);
      membresias[opciones[Math.floor(rng() * opciones.length)]].push(u);
    }
    const aristas = [];
    for (const [u, v] of AGM.pares(n)) {
      if (rng() < AGM.probabilidad(u, v, membresias, new Array(K).fill(p), eps).p) aristas.push([u, v]);
    }
    return { n, membresias, aristas };
  },
};

// ------------------------------------------------------------------
// BigCLAM
// ------------------------------------------------------------------
const G_MAX = 10;
const F_MAX = 1000;
const EPS_X = -Math.log1p(-1e-8);

const BigCLAM = {
  objetivoFila(x, Yvec, resto) {
    let l = 0;
    for (const y of Yvec) l += log1menosExp(punto(x, y) + EPS_X);
    return l - punto(x, resto);
  },

  gradienteFila(x, Yvec, resto) {
    let g = escalarVec(resto, -1);
    for (const y of Yvec) {
      const d = Math.max(punto(x, y) + EPS_X, 1e-12);
      g = sumaVec(g, escalarVec(y, 1 / Math.expm1(d)));
    }
    return g.map((v) => Math.max(-G_MAX, Math.min(G_MAX, v)));
  },

  // búsqueda lineal por retroceso con condición de Armijo; devuelve los intentos
  busquedaLineal(x, g, Yvec, resto, alfa = 0.05, beta = 0.3, intentos = 15) {
    const l0 = BigCLAM.objetivoFila(x, Yvec, resto);
    const pruebas = [];
    let t = 1;
    for (let i = 0; i < intentos; i++) {
      const nuevo = x.map((v, c) => Math.min(F_MAX, Math.max(0, v + t * g[c])));
      const l = BigCLAM.objetivoFila(nuevo, Yvec, resto);
      const umbral = l0 + alfa * punto(g, restaVec(nuevo, x));
      const acepta = l >= umbral;
      pruebas.push({ t, l, umbral, acepta, nuevo });
      if (acepta) return { l0, pruebas, nuevo };
      t *= beta;
    }
    return { l0, pruebas, nuevo: x.slice() };
  },

  conductancia(vecinos, conjunto, volTotal) {
    let corte = 0, vol = 0;
    for (const u of conjunto) {
      vol += vecinos[u].length;
      for (const v of vecinos[u]) if (!conjunto.has(v)) corte++;
    }
    return corte / Math.max(1, Math.min(vol, volTotal - vol));
  },

  inicializar(vecinos, K, rng) {
    const n = vecinos.length;
    const volTotal = d3.sum(vecinos, (v) => v.length);
    const phi = vecinos.map((v, u) => BigCLAM.conductancia(vecinos, new Set([u, ...v]), volTotal));
    let minimos = d3.range(n).filter((u) => vecinos[u].every((v) => phi[u] <= phi[v]));
    minimos = minimos.sort((a, b) => phi[a] - phi[b]).slice(0, K);
    const resto = permutar(n, rng).filter((u) => !minimos.includes(u));
    const semillas = minimos.concat(resto.slice(0, K - minimos.length));
    const F = d3.range(n).map(() => ceros(K));
    semillas.forEach((u, k) => [u, ...vecinos[u]].forEach((v) => (F[v][k] = 1)));
    return { F, semillas, phi };
  },

  logVerosimilitud(F, aristas) {
    const n = F.length, K = F[0].length;
    let l = 0, dE = 0;
    for (const [u, v] of aristas) {
      const d = punto(F[u], F[v]) + EPS_X;
      l += log1menosExp(d);
      dE += d;
    }
    const S = F.reduce((s, f) => sumaVec(s, f), ceros(K));
    const todos = (punto(S, S) - d3.sum(F, (f) => punto(f, f))) / 2 + (EPS_X * n * (n - 1)) / 2;
    return l - (todos - dE);
  },

  // ajuste completo (sin animación), igual que bigclam.ajustar en Python
  ajustar(vecinos, K, rng, iterMax = 100) {
    const { F } = BigCLAM.inicializar(vecinos, K, rng);
    let S = F.reduce((s, f) => sumaVec(s, f), ceros(K));
    for (let it = 0; it < iterMax; it++) {
      for (const u of permutar(F.length, rng)) {
        const Yvec = vecinos[u].map((v) => F[v]);
        const resto = restaVec(restaVec(S, F[u]), Yvec.reduce((s, y) => sumaVec(s, y), ceros(K)));
        const g = BigCLAM.gradienteFila(F[u], Yvec, resto);
        const nuevo = BigCLAM.busquedaLineal(F[u], g, Yvec, resto).nuevo;
        S = sumaVec(S, restaVec(nuevo, F[u]));
        F[u] = nuevo;
      }
    }
    return F;
  },

  umbral(n, m) {
    const eps = (2 * m) / (n * (n - 1));
    return Math.sqrt(-Math.log1p(-eps));
  },
};

// ------------------------------------------------------------------
// CoDA: AGM dirigido con F (salida) y H (entrada)
// ------------------------------------------------------------------
const SEMILLA_CODA = 1;

const CoDA = {
  logVerosimilitud(F, H, aristas) {
    const n = F.length, K = F[0].length;
    let l = 0, dE = 0;
    for (const [u, v] of aristas) {
      const d = punto(F[u], H[v]) + EPS_X;
      l += log1menosExp(d);
      dE += d;
    }
    const SF = F.reduce((s, f) => sumaVec(s, f), ceros(K));
    const SH = H.reduce((s, h) => sumaVec(s, h), ceros(K));
    const todos = punto(SF, SH) - d3.sum(F, (f, u) => punto(f, H[u])) + EPS_X * n * (n - 1);
    return l - (todos - dE);
  },

  // actualiza todas las filas de X (F o H) con la otra matriz Y fija
  mediaIteracion(X, Y, vecinosDe, rng) {
    const K = X[0].length;
    const SY = Y.reduce((s, y) => sumaVec(s, y), ceros(K));
    for (const u of permutar(X.length, rng)) {
      const Yvec = vecinosDe[u].map((v) => Y[v]);
      const resto = restaVec(restaVec(SY, Y[u]), Yvec.reduce((s, y) => sumaVec(s, y), ceros(K)));
      const g = BigCLAM.gradienteFila(X[u], Yvec, resto);
      X[u] = BigCLAM.busquedaLineal(X[u], g, Yvec, resto).nuevo;
    }
  },

  // red de ejemplo: fans -> famosos (dos modos) y amigos <-> amigos (cohesiva)
  ejemplo(semilla = SEMILLA_CODA) {
    const rng = generador(semilla);
    const fans = d3.range(0, 10), famosos = d3.range(10, 14), amigos = d3.range(14, 22);
    const amigosTodos = amigos.concat([0, 1, 2]);
    const n = 22, aristas = [];
    for (let u = 0; u < n; u++) {
      for (let v = 0; v < n; v++) {
        if (u === v) continue;
        let q = 1 - 0.01;
        if (fans.includes(u) && famosos.includes(v)) q *= 1 - 0.55;
        if (amigosTodos.includes(u) && amigosTodos.includes(v)) q *= 1 - 0.45;
        if (rng() < 1 - q) aristas.push([u, v]);
      }
    }
    const grupo = (u) => (fans.includes(u) ? (u <= 2 ? "fan y amigo" : "fan") : famosos.includes(u) ? "famoso" : "amigo");
    return { n, aristas, grupo, fans, famosos, amigos };
  },
};

// ------------------------------------------------------------------
// NMF bayesiana con ARD (Psorakis et al., 2011)
// ------------------------------------------------------------------
const BNMF = {
  // estado: V = A + diag(grado), W (n x K), H (K x n), beta (K)
  iniciar(n, aristas, K, rng) {
    const V = d3.range(n).map(() => ceros(n));
    aristas.forEach(([u, v]) => { V[u][v] = 1; V[v][u] = 1; });
    V.forEach((fila, u) => (fila[u] = d3.sum(fila)));
    return {
      n, K, V, beta: new Array(K).fill(1),
      W: d3.range(n).map(() => d3.range(K).map(rng)),
      H: d3.range(K).map(() => d3.range(n).map(rng)),
      iter: 0, vivas: [K],
    };
  },

  paso(e, a = 1, b = 2) {
    const { n, K, V, W, H } = e;
    const tiny = 1e-12;
    const producto = () => d3.range(n).map((i) => d3.range(n).map((j) => {
      let s = 0;
      for (let k = 0; k < K; k++) s += W[i][k] * H[k][j];
      return V[i][j] / (s + tiny);
    }));
    let R = producto();
    const sumaW = d3.range(K).map((k) => d3.sum(W, (w) => w[k]));
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < n; j++) {
        let num = 0;
        for (let i = 0; i < n; i++) num += W[i][k] * R[i][j];
        H[k][j] = (H[k][j] / (sumaW[k] + e.beta[k] * H[k][j] + tiny)) * num;
      }
    }
    R = producto();
    const sumaH = H.map((h) => d3.sum(h));
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) {
        let num = 0;
        for (let j = 0; j < n; j++) num += R[i][j] * H[k][j];
        W[i][k] = (W[i][k] / (sumaH[k] + W[i][k] * e.beta[k] + tiny)) * num;
      }
    }
    e.beta = d3.range(K).map((k) =>
      (n + a - 1) / (0.5 * (d3.sum(W, (w) => w[k] ** 2) + d3.sum(H[k], (h) => h ** 2)) + b));
    e.iter++;
    e.vivas.push(BNMF.activas(e).filter(Boolean).length);
  },

  pesos(e) {
    return d3.range(e.K).map((k) => d3.sum(e.W, (w) => w[k]) * d3.sum(e.H[k]));
  },

  activas(e, tol = 0.02) {
    const p = BNMF.pesos(e);
    const max = d3.max(p);
    return p.map((x) => x > tol * max);
  },
};
