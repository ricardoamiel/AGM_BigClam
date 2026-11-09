// Pestaña 5: propuestas recientes (Ego-splitting, NMF bayesiana, NOCD).
"use strict";

document.querySelectorAll(".opcion").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".opcion").forEach((x) => x.classList.toggle("activa", x === b));
    document.querySelectorAll(".modo").forEach((m) => m.classList.toggle("activo", m.id === "modo-" + b.dataset.modo));
  })
);

function componentesConexas(nodos, aristas) {
  const vecinos = new Map(nodos.map((u) => [u, []]));
  aristas.forEach(([u, v]) => { vecinos.get(u).push(v); vecinos.get(v).push(u); });
  const visto = new Set(), partes = [];
  for (const u of nodos) {
    if (visto.has(u)) continue;
    const parte = [], pila = [u];
    visto.add(u);
    while (pila.length) {
      const x = pila.pop();
      parte.push(x);
      for (const y of vecinos.get(x)) if (!visto.has(y)) { visto.add(y); pila.push(y); }
    }
    partes.push(parte.sort());
  }
  return partes;
}

// ==================================================================
// Ego-splitting: ejemplo de la figura 1 del paper
// ==================================================================
const ego = {
  nodos: ["a", "b", "c", "d", "e", "f", "g", "h"],
  aristas: [["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"], ["c", "e"], ["c", "f"], ["d", "e"], ["d", "f"],
    ["e", "f"], ["f", "g"], ["f", "h"], ["g", "h"]],
  pos: { a: [60, 90], b: [60, 240], c: [170, 165], d: [265, 80], e: [265, 250], f: [360, 165], g: [465, 90], h: [465, 240] },
  paso: 0,
};
ego.vecinos = (u) => ego.aristas.filter((a) => a.includes(u)).map(([x, y]) => (x === u ? y : x));

// partición de cada ego-net con componentes conexas y asignación de personas
ego.partes = {};
ego.persona = {};
ego.nodos.forEach((u) => {
  const N = ego.vecinos(u);
  const internas = ego.aristas.filter(([x, y]) => N.includes(x) && N.includes(y));
  ego.partes[u] = componentesConexas(N, internas);
  ego.persona[u] = {};
  ego.partes[u].forEach((parte, i) => parte.forEach((v) => (ego.persona[u][v] = u + (i + 1))));
});
ego.personas = ego.nodos.flatMap((u) => ego.partes[u].map((_, i) => u + (i + 1)));
ego.aristasPersonas = ego.aristas.map(([u, v]) => [ego.persona[u][v], ego.persona[v][u]]);
ego.grupos = componentesConexas(ego.personas, ego.aristasPersonas);
ego.comunidades = ego.grupos.map((g) => [...new Set(g.map((p) => p[0]))]);
ego.total = 1 + ego.nodos.length + 3;

function egoPosPersona(p) {
  // la persona se dibuja desplazada hacia los vecinos que ve
  const u = p[0], i = +p.slice(1) - 1;
  const [x, y] = ego.pos[u];
  if (ego.partes[u].length === 1) return [x, y];
  const parte = ego.partes[u][i];
  const cx = d3.mean(parte, (v) => ego.pos[v][0]), cy = d3.mean(parte, (v) => ego.pos[v][1]);
  const d = Math.hypot(cx - x, cy - y) || 1;
  return [x + ((cx - x) / d) * 34, y + ((cy - y) / d) * 34];
}

function egoNodo(svg, x, y, texto, relleno, r = 17, borde = null) {
  svg.append("circle").attr("cx", x).attr("cy", y).attr("r", r).attr("fill", relleno)
    .attr("stroke", borde || variable("--texto")).attr("stroke-width", borde ? 3 : 1);
  svg.append("text").attr("x", x).attr("y", y + 5).attr("text-anchor", "middle").attr("font-weight", 700)
    .attr("font-size", 14).attr("fill", "#fff").text(texto);
}

function egoDibujar() {
  const p = ego.paso;
  const izq = d3.select("#egoRed").html(""), der = d3.select("#egoLocal").html("");
  const linea = (svg, a, b, col, w = 2, dash = null) =>
    svg.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1])
      .attr("stroke", col).attr("stroke-width", w).attr("stroke-dasharray", dash);
  const etapas = ["grafo", "ego-nets locales", "grafo de personas", "partición global", "resultado"];
  const etapa = p === 0 ? 0 : p <= ego.nodos.length ? 1 : p - ego.nodos.length + 1;
  $("egoProgreso").innerHTML = etapas.map((e, i) => `<span class="${i === etapa ? "activo" : ""}">${i + 1}. ${e}</span>`).join("");

  if (p <= ego.nodos.length) {
    // grafo original y ego-net del nodo actual
    const u = p > 0 ? ego.nodos[p - 1] : null;
    const N = u ? ego.vecinos(u) : [];
    $("egoTitulo1").textContent = "Grafo original";
    $("egoTitulo2").textContent = u ? `Ego-net de ${u}: sus vecinos, sin ${u}` : "Ego-net del nodo actual";
    ego.aristas.forEach(([a, b]) => linea(izq, ego.pos[a], ego.pos[b], variable("--arista"), 2));
    ego.nodos.forEach((v, i) => {
      const hecho = p > 0 && i < p - 1;
      const col = v === u ? variable("--acento2") : N.includes(v) ? color(0) : variable("--suave");
      egoNodo(izq, ...ego.pos[v], v, col, 17, v === u ? variable("--resalte") : null);
      if (hecho || v === u) {
        const k = ego.partes[v].length;
        izq.append("text").attr("x", ego.pos[v][0] + 20).attr("y", ego.pos[v][1] - 16).attr("font-size", 12)
          .attr("font-weight", 700).attr("fill", k > 1 ? color(2) : variable("--suave"))
          .text(k > 1 ? `${k} personas` : "1 persona");
      }
    });
    if (u) {
      const escala = (q) => [80 + (q[0] - 60) * 0.85, 30 + (q[1] - 60) * 1.05];
      der.append("circle").attr("cx", escala(ego.pos[u])[0]).attr("cy", escala(ego.pos[u])[1]).attr("r", 17)
        .attr("fill", "none").attr("stroke", variable("--suave")).attr("stroke-dasharray", "5 4");
      der.append("text").attr("x", escala(ego.pos[u])[0]).attr("y", escala(ego.pos[u])[1] + 5)
        .attr("text-anchor", "middle").attr("fill", variable("--suave")).text(u);
      ego.aristas.filter(([a, b]) => N.includes(a) && N.includes(b))
        .forEach(([a, b]) => linea(der, escala(ego.pos[a]), escala(ego.pos[b]), variable("--texto"), 2.5));
      ego.partes[u].forEach((parte, i) =>
        parte.forEach((v) => egoNodo(der, ...escala(ego.pos[v]), v, color(i))));
    }
    const exp = $("egoExp");
    if (!u) {
      exp.innerHTML = `<span class="etapa">Idea</span>
        <p>Ego-splitting no inventa un modelo nuevo: usa un algoritmo de partición dos veces. Primero, en el
        vecindario de cada nodo; después, en un grafo donde cada nodo se partió en sus personas. Así una
        partición rápida (componentes conexas, Louvain, propagación de etiquetas) produce comunidades superpuestas.</p>
        <p>El ejemplo es la figura 1 del paper: tres comunidades {a, b, c}, {c, d, e, f} y {f, g, h}.</p>`;
    } else {
      const k = ego.partes[u].length;
      exp.innerHTML = `<span class="etapa">Paso local: nodo ${u}</span>
        <p>La ego-net de ${u} son sus vecinos {${N.join(", ")}} y las aristas entre ellos, sin ${u}.
        Al quitar ${u}, sus grupos dejan de estar unidos por él.</p>
        <p>Componentes conexas: ${ego.partes[u].map((q, i) => `${u}${i + 1} ve {${q.join(", ")}}`).join("; ")}.</p>
        <p>${k > 1 ? `<span class="ok">${u} se parte en ${k} personas:</span> pertenece a ${k} contextos distintos.`
          : `${u} conserva una sola persona.`}</p>`;
    }
    return;
  }

  const fase = p - ego.nodos.length;   // 1: personas, 2: global, 3: resultado
  if (fase <= 2) {
    $("egoTitulo1").textContent = fase === 1 ? "Grafo de personas" : "Partición global del grafo de personas";
    $("egoTitulo2").textContent = "Personas de cada nodo";
    const grupoDe = new Map();
    ego.grupos.forEach((g, i) => g.forEach((q) => grupoDe.set(q, i)));
    ego.aristasPersonas.forEach(([a, b]) => linea(izq, egoPosPersona(a), egoPosPersona(b), variable("--arista"), 2));
    ego.personas.forEach((q) => egoNodo(izq, ...egoPosPersona(q), q,
      fase === 2 ? color(grupoDe.get(q)) : variable("--suave"), 16));
    ego.nodos.forEach((u, i) => {
      const y = 28 + i * 37;
      der.append("text").attr("x", 30).attr("y", y + 5).attr("font-weight", 700).attr("fill", variable("--texto")).text(u);
      ego.partes[u].forEach((parte, j) => {
        const q = u + (j + 1);
        egoNodo(der, 90 + j * 190, y, q, fase === 2 ? color(grupoDe.get(q)) : variable("--suave"), 14);
        der.append("text").attr("x", 110 + j * 190).attr("y", y + 5).attr("font-size", 13)
          .attr("fill", variable("--texto")).text(`ve {${parte.join(", ")}}`);
      });
    });
    $("egoExp").innerHTML = fase === 1
      ? `<span class="etapa">Grafo de personas</span>
         <p>Cada arista (u, v) del grafo original une la persona de u que ve a v con la persona de v que ve a u.
         Por ejemplo (c, d) pasa a ser (${ego.persona.c.d}, ${ego.persona.d.c}). Las personas de c y de f ya no
         comparten aristas: el grafo quedó separado en sus contextos.</p>`
      : `<span class="etapa">Paso global</span>
         <p>Se particiona el grafo de personas, aquí otra vez con componentes conexas.
         Salen ${ego.grupos.length} grupos: ${ego.grupos.map((g) => "{" + g.join(", ") + "}").join(", ")}.</p>`;
    return;
  }

  $("egoTitulo1").textContent = "Resultado: comunidades superpuestas";
  $("egoTitulo2").textContent = "Cada grupo de personas, traducido a nodos";
  ego.aristas.forEach(([a, b]) => linea(izq, ego.pos[a], ego.pos[b], variable("--arista"), 2));
  ego.nodos.forEach((u) => {
    const pesos = ego.comunidades.map((c) => (c.includes(u) ? 1 : 0));
    const g = torta(izq, ...ego.pos[u], 17, pesos);
    izq.append("text").attr("x", ego.pos[u][0]).attr("y", ego.pos[u][1] + 5).attr("text-anchor", "middle")
      .attr("font-weight", 700).attr("fill", "#fff").text(u);
    if (d3.sum(pesos) > 1) g.append("circle").attr("r", 23).attr("fill", "none").attr("stroke", variable("--resalte")).attr("stroke-width", 3);
  });
  ego.comunidades.forEach((c, i) => {
    der.append("rect").attr("x", 30).attr("y", 40 + i * 70).attr("width", 26).attr("height", 26).attr("rx", 5).attr("fill", color(i));
    der.append("text").attr("x", 70).attr("y", 60 + i * 70).attr("font-size", 18).attr("fill", variable("--texto"))
      .text(`{${ego.grupos[i].join(", ")}}  →  {${c.join(", ")}}`);
  });
  $("egoExp").innerHTML = `<span class="etapa">Resultado</span>
    <p>c y f aparecen en dos comunidades porque tienen dos personas cada uno. Se recuperan exactamente las tres
    comunidades del paper sin fijar K y con dos pasadas de un algoritmo de partición.</p>
    <p>Fortaleza: escala a decenas de miles de millones de aristas y hereda la velocidad del algoritmo base.
    Límite: si dos comunidades comparten varios nodos, las ego-nets de esos nodos quedan conectadas y las
    componentes conexas no las separan; por eso en grafos reales el paso local usa Louvain o propagación de etiquetas.</p>`;
}

let egoTemporizador = null;
function egoReproducir() {
  if (egoTemporizador) {
    clearInterval(egoTemporizador); egoTemporizador = null;
    $("egoPlay").textContent = "Reproducir";
    return;
  }
  $("egoPlay").textContent = "Pausa";
  egoTemporizador = setInterval(() => {
    if (ego.paso >= ego.total - 1) return egoReproducir();
    ego.paso++; egoDibujar();
  }, 1300);
}

// ==================================================================
// NMF bayesiana: la prior ARD apaga comunidades
// ==================================================================
const bn = {};

function bnReiniciar() {
  const solape = +$("bnSolape").value;
  $("bnSolapev").textContent = solape.toFixed(1);
  const semilla = +$("bnSemilla").value;
  const g = AGM.plantado(24, 3, solape, 0.55, 0.02, semilla);
  bn.g = g;
  bn.pos = distribuir(g.n, g.aristas, 440, 380);
  bn.e = BNMF.iniciar(g.n, g.aristas, +$("bnK").value, generador(semilla));
  bnDibujar();
}

function bnDibujar() {
  const e = bn.e;
  const pesos = BNMF.pesos(e), vivas = BNMF.activas(e);
  const total = d3.sum(pesos);
  const svg = d3.select("#bnRed").html("");
  bn.g.aristas.forEach(([a, b]) =>
    svg.append("line").attr("x1", bn.pos[a][0]).attr("y1", bn.pos[a][1]).attr("x2", bn.pos[b][0]).attr("y2", bn.pos[b][1])
      .attr("stroke", variable("--arista")).attr("stroke-width", 1.1));
  // color por comunidad viva en orden de peso
  const orden = d3.range(e.K).filter((k) => vivas[k]).sort((a, b) => pesos[b] - pesos[a]);
  const colorDe = (k) => (orden.indexOf(k) >= 0 && orden.indexOf(k) < 4 ? color(orden.indexOf(k)) : variable("--suave"));
  bn.pos.forEach(([x, y], u) => {
    const w = e.W[u].map((v, k) => (vivas[k] ? v : 0));
    const s = d3.sum(w) || 1;
    const arco = d3.arc().innerRadius(0).outerRadius(11);
    const g = svg.append("g").attr("transform", `translate(${x},${y})`);
    g.selectAll("path").data(d3.pie().sort(null)(w.map((v) => v / s))).join("path").attr("d", arco)
      .attr("fill", (d, k) => colorDe(k));
    g.append("circle").attr("r", 11).attr("fill", "none").attr("stroke", variable("--texto")).attr("stroke-width", 0.8);
  });

  const barras = d3.select("#bnBarras").html("");
  const x = d3.scaleBand().domain(d3.range(e.K)).range([50, 410]).padding(0.2);
  const y = d3.scaleLinear().domain([0, 1]).range([250, 15]);
  ejes(barras, x, y, 250, 50, "comunidad k", "fracción de W H");
  barras.selectAll("rect.b").data(pesos).join("rect").attr("class", "b")
    .attr("x", (d, k) => x(k)).attr("width", x.bandwidth()).attr("y", (d) => y(d / total))
    .attr("height", (d) => 250 - y(d / total)).attr("fill", (d, k) => (vivas[k] ? colorDe(k) : variable("--borde")));

  const curva = d3.select("#bnCurva").html("");
  const xc = d3.scaleLinear().domain([0, Math.max(20, e.vivas.length - 1)]).range([50, 400]);
  const yc = d3.scaleLinear().domain([0, e.K]).range([170, 10]);
  ejes(curva, xc, yc, 170, 50, "iteración", "vivas");
  curva.append("path").datum(e.vivas).attr("fill", "none").attr("stroke", color(1)).attr("stroke-width", 2.5)
    .attr("d", d3.line().x((d, i) => xc(i)).y((d) => yc(d)));
  curva.append("line").attr("x1", 50).attr("x2", 400).attr("y1", yc(3)).attr("y2", yc(3))
    .attr("stroke", color(2)).attr("stroke-dasharray", "6 5");

  $("bnExp").innerHTML = `<span class="etapa">Iteración ${e.iter}</span>
    <p>Comunidades vivas: <b>${vivas.filter(Boolean).length}</b> de ${e.K} (línea verde: 3 reales).</p>
    <div class="formula">β<sub>k</sub> ← (N + a − 1) / (½ Σ w<sub>ik</sub>² + ½ Σ h<sub>kj</sub>² + b)</div>
    <p>Cada comunidad tiene una precisión β<sub>k</sub>. Si explica pocos datos, sus pesos bajan, β<sub>k</sub> sube
    y la prior la empuja a cero: es la determinación automática de relevancia (ARD). No hace falta probar varios K.</p>
    <p>Pruebe subir el solape: el modelo tiende a crear una comunidad propia para cada zona solapada, porque una
    factorización aditiva explica mejor ese bloque denso con un componente extra.</p>`;
}

let bnTemporizador = null;
function bnReproducir() {
  if (bnTemporizador) {
    clearInterval(bnTemporizador); bnTemporizador = null;
    $("bnPlay").textContent = "Reproducir";
    return;
  }
  $("bnPlay").textContent = "Pausa";
  bnTemporizador = setInterval(() => {
    if (bn.e.iter >= 400) return bnReproducir();
    for (let i = 0; i < 5; i++) BNMF.paso(bn.e);
    bnDibujar();
  }, 120);
}

// ==================================================================
// NOCD: entrenamiento precalculado en Python
// ==================================================================
const nc = { idx: 0 };

function ncArquitectura() {
  const svg = d3.select("#ncArq").html("");
  const cajas = [["A y X", "grafo y atributos"], ["GCN capa 1", "128 + ReLU + BatchNorm"], ["GCN capa 2", "K salidas + ReLU"],
    ["F ≥ 0", "membresías N × K"], ["Decodificador", "p(u, v) = 1 − e^(−Fu·Fv)"], ["Pérdida", "promedio balanceado"]];
  cajas.forEach(([t, s], i) => {
    const x = 10 + i * 165;
    svg.append("rect").attr("x", x).attr("y", 30).attr("width", 145).attr("height", 80).attr("rx", 10)
      .attr("fill", i === 4 ? d3.color(color(1)).copy({ opacity: 0.18 }) : variable("--fondo"))
      .attr("stroke", i === 4 ? color(1) : color(0)).attr("stroke-width", 2);
    svg.append("text").attr("x", x + 72).attr("y", 62).attr("text-anchor", "middle").attr("font-weight", 700)
      .attr("fill", variable("--texto")).text(t);
    svg.append("text").attr("x", x + 72).attr("y", 86).attr("text-anchor", "middle").attr("font-size", 11)
      .attr("fill", variable("--suave")).text(s);
    if (i < cajas.length - 1)
      svg.append("line").attr("x1", x + 147).attr("x2", x + 163).attr("y1", 70).attr("y2", 70)
        .attr("stroke", variable("--suave")).attr("stroke-width", 2);
  });
  svg.append("text").attr("x", 500).attr("y", 138).attr("text-anchor", "middle").attr("font-size", 12)
    .attr("fill", color(1)).text("el decodificador y la pérdida son exactamente los de BigCLAM; cambia quién produce F");
}

function ncDibujar() {
  if (typeof DATOS_NOCD === "undefined") return;
  const D = DATOS_NOCD, t = D.traza[nc.idx];
  $("ncEpoca").max = D.traza.length - 1;
  $("ncEpocav").textContent = t.epoca;
  ncArquitectura();
  const x = d3.scaleLinear().domain(d3.extent(D.pos, (p) => p[0])).range([28, 412]);
  const y = d3.scaleLinear().domain(d3.extent(D.pos, (p) => p[1])).range([28, 352]);
  const svg = d3.select("#ncRed").html("");
  D.aristas.forEach(([a, b]) =>
    svg.append("line").attr("x1", x(D.pos[a][0])).attr("y1", y(D.pos[a][1])).attr("x2", x(D.pos[b][0])).attr("y2", y(D.pos[b][1]))
      .attr("stroke", variable("--arista")).attr("stroke-width", 1.1));
  D.pos.forEach((p, u) => {
    torta(svg, x(p[0]), y(p[1]), 11, t.F[u]);
    svg.append("text").attr("x", x(p[0])).attr("y", y(p[1]) + 4).attr("text-anchor", "middle").attr("font-size", 9)
      .attr("font-weight", 700).attr("fill", "#fff").text(u);
  });

  const curva = d3.select("#ncCurva").html("");
  const xc = d3.scaleLinear().domain(d3.extent(D.traza, (d) => d.epoca)).range([50, 400]);
  const yc = d3.scaleLog().domain(d3.extent(D.traza, (d) => d.perdida)).range([210, 10]);
  ejes(curva, xc, yc, 210, 50, "época", "pérdida (escala log)");
  curva.append("path").datum(D.traza).attr("fill", "none").attr("stroke", color(0)).attr("stroke-width", 2.5)
    .attr("d", d3.line().x((d) => xc(d.epoca)).y((d) => yc(d.perdida)));
  curva.append("circle").attr("cx", xc(t.epoca)).attr("cy", yc(t.perdida)).attr("r", 6).attr("fill", color(1));

  const conMembresia = t.F.filter((f) => d3.max(f) >= 0.5).length;
  const enDos = t.F.filter((f) => f.filter((v) => v >= 0.5).length >= 2).length;
  $("ncExp").innerHTML = `<span class="etapa">Época ${t.epoca}</span>
    <p>Pérdida: ${fmt3(t.perdida)}. Nodos con alguna F ≥ 0.5: ${conMembresia} de ${D.n}; en dos o más: ${enDos}.</p>
    <p>BigCLAM optimiza cada fila F<sub>u</sub> por separado, así que necesita reentrenar para un nodo nuevo.
    NOCD aprende una función F = GCN(A, X): los pesos de la red se comparten entre nodos, puede usar atributos
    X y se entrena por descenso de gradiente estocástico en GPU.</p>
    <p>Comunidades plantadas: ${D.verdad.map((c, i) => `${NOMBRES[i]} = {${c.join(", ")}}`).join("; ")}.</p>`;
}

let ncTemporizador = null;
function ncReproducir() {
  if (ncTemporizador) {
    clearInterval(ncTemporizador); ncTemporizador = null;
    $("ncPlay").textContent = "Reproducir entrenamiento";
    return;
  }
  $("ncPlay").textContent = "Pausa";
  if (nc.idx >= DATOS_NOCD.traza.length - 1) nc.idx = 0;
  ncTemporizador = setInterval(() => {
    if (nc.idx >= DATOS_NOCD.traza.length - 1) return ncReproducir();
    nc.idx++; $("ncEpoca").value = nc.idx; ncDibujar();
  }, 250);
}

// eventos
$("egoPaso").addEventListener("click", () => { ego.paso = Math.min(ego.paso + 1, ego.total - 1); egoDibujar(); });
$("egoAtras").addEventListener("click", () => { ego.paso = Math.max(ego.paso - 1, 0); egoDibujar(); });
$("egoReset").addEventListener("click", () => { ego.paso = 0; egoDibujar(); });
$("egoPlay").addEventListener("click", egoReproducir);
$("bnPaso").addEventListener("click", () => { for (let i = 0; i < 10; i++) BNMF.paso(bn.e); bnDibujar(); });
$("bnPlay").addEventListener("click", bnReproducir);
$("bnReset").addEventListener("click", bnReiniciar);
["bnSolape", "bnK", "bnSemilla"].forEach((id) => $(id).addEventListener("change", bnReiniciar));
$("bnSolape").addEventListener("input", () => ($("bnSolapev").textContent = (+$("bnSolape").value).toFixed(1)));
$("ncEpoca").addEventListener("input", () => { nc.idx = +$("ncEpoca").value; ncDibujar(); });
$("ncPlay").addEventListener("click", ncReproducir);
$("tema").addEventListener("click", () => { egoDibujar(); bnDibujar(); ncDibujar(); });

egoDibujar();
bnReiniciar();
ncDibujar();
