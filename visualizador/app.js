// Interfaz del visualizador: cuatro pestañas que muestran cada paso de los modelos.
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = d3.format(".2f");
const fmt3 = d3.format(".3f");
const vec = (v) => "(" + v.map(fmt).join(", ") + ")";
const NOMBRES = ["A", "B", "C", "D"];

function color(i) {
  return getComputedStyle(document.documentElement).getPropertyValue("--c" + i).trim();
}
function variable(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();
}

// nodo dibujado como torta según sus pesos por comunidad
function torta(g, x, y, r, pesos) {
  const total = d3.sum(pesos);
  const nodo = g.append("g").attr("transform", `translate(${x},${y})`);
  if (total < 1e-9) {
    nodo.append("circle").attr("r", r).attr("fill", variable("--vacio")).attr("stroke", variable("--suave"));
    return nodo;
  }
  const arco = d3.arc().innerRadius(0).outerRadius(r);
  const datos = d3.pie().sort(null)(pesos.map((p) => Math.max(p, 0)));
  nodo.selectAll("path").data(datos).join("path").attr("d", arco).attr("fill", (d, i) => color(i));
  nodo.append("circle").attr("r", r).attr("fill", "none").attr("stroke", variable("--texto")).attr("stroke-width", 0.8);
  return nodo;
}

// distribución con fuerzas, calculada de una vez para que no se mueva
function distribuir(n, aristas, ancho, alto, margen = 26) {
  const nodos = d3.range(n).map((i) => ({ id: i }));
  const enlaces = aristas.map(([s, t]) => ({ source: s, target: t }));
  d3.forceSimulation(nodos)
    .force("carga", d3.forceManyBody().strength(-120))
    .force("enlace", d3.forceLink(enlaces).distance(40))
    .force("centro", d3.forceCenter(ancho / 2, alto / 2))
    .force("colision", d3.forceCollide(14))
    .stop()
    .tick(400);
  const x = d3.scaleLinear().domain(d3.extent(nodos, (d) => d.x)).range([margen, ancho - margen]);
  const y = d3.scaleLinear().domain(d3.extent(nodos, (d) => d.y)).range([margen, alto - margen]);
  return nodos.map((d) => [x(d.x), y(d.y)]);
}

function ejes(svg, x, y, alto, izq, etiquetaX, etiquetaY) {
  svg.append("g").attr("class", "eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(6));
  svg.append("g").attr("class", "eje").attr("transform", `translate(${izq},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("x", (x.range()[0] + x.range()[1]) / 2).attr("y", alto + 34)
    .attr("text-anchor", "middle").attr("fill", variable("--suave")).attr("font-size", 12).text(etiquetaX);
  svg.append("text").attr("transform", `translate(14,${(y.range()[0] + y.range()[1]) / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").attr("fill", variable("--suave")).attr("font-size", 12).text(etiquetaY);
}

// ==================================================================
// Pestaña 1: proceso generativo del AGM
// ==================================================================
const agm = {
  membresias: [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [7, 8, 9]],
  n: 10,
  pos: [[40, 120], [95, 50], [150, 110], [140, 200], [215, 190], [255, 290], [320, 230], [340, 300], [430, 260], [470, 170]],
};

function agmParametros() {
  return { p: ["pA", "pB", "pC"].map((id) => +$(id).value), eps: +$("eps").value, semilla: +$("semillaAGM").value };
}

function agmReiniciar() {
  const { semilla } = agmParametros();
  agm.rng = generador(semilla);
  agm.pares = permutar(45, generador(semilla + 1)).map((i) => AGM.pares(10)[i]);
  agm.idx = 0;
  agm.aristas = [];
  agm.historial = [];
  agmDibujar();
}

function agmPaso() {
  if (agm.idx >= agm.pares.length) return false;
  const { p, eps } = agmParametros();
  const [u, v] = agm.pares[agm.idx++];
  const { p: prob, compartidas } = AGM.probabilidad(u, v, agm.membresias, p, eps);
  const r = agm.rng();
  const agrega = r < prob;
  if (agrega) agm.aristas.push([u, v]);
  agm.historial.push({ u, v, prob, compartidas, r, agrega });
  return true;
}

// red completa para la pestaña de máxima verosimilitud
function agmCompleto() {
  const { p, eps, semilla } = agmParametros();
  const rng = generador(semilla);
  const pares = permutar(45, generador(semilla + 1)).map((i) => AGM.pares(10)[i]);
  return pares.filter(([u, v]) => rng() < AGM.probabilidad(u, v, agm.membresias, p, eps).p);
}

function agmDibujar() {
  const { p } = agmParametros();
  const actual = agm.historial[agm.historial.length - 1];
  const pesos = d3.range(agm.n).map((u) => agm.membresias.map((m) => (m.includes(u) ? 1 : 0)));

  // grafo bipartito
  const bip = d3.select("#agmBip").html("");
  const xs = d3.range(agm.n).map((u) => 40 + u * 49);
  const cx = [xs[2], xs[5], xs[8]];
  agm.membresias.forEach((nodos, c) =>
    nodos.forEach((u) => {
      const resalta = actual && actual.compartidas.includes(c) && (u === actual.u || u === actual.v);
      bip.append("line").attr("x1", cx[c]).attr("y1", 90).attr("x2", xs[u]).attr("y2", 245)
        .attr("stroke", color(c)).attr("stroke-width", resalta ? 5 : 1.8).attr("opacity", actual && !resalta ? 0.45 : 1);
    })
  );
  agm.membresias.forEach((_, c) => {
    const g = bip.append("g").attr("transform", `translate(${cx[c]},60)`);
    g.append("rect").attr("x", -38).attr("y", -30).attr("width", 76).attr("height", 60).attr("rx", 6)
      .attr("fill", variable("--tarjeta")).attr("stroke", color(c)).attr("stroke-width", 2.5);
    g.append("text").attr("text-anchor", "middle").attr("y", -4).attr("font-weight", 700).attr("font-size", 20)
      .attr("fill", color(c)).text(NOMBRES[c]);
    g.append("text").attr("text-anchor", "middle").attr("y", 18).attr("font-size", 13)
      .attr("fill", variable("--texto")).text("p = " + fmt(p[c]));
  });
  xs.forEach((x, u) => {
    if (actual && (u === actual.u || u === actual.v))
      bip.append("circle").attr("cx", x).attr("cy", 260).attr("r", 21).attr("fill", variable("--resalte"));
    torta(bip, x, 260, 15, pesos[u]);
    bip.append("text").attr("x", x).attr("y", 300).attr("text-anchor", "middle").attr("font-weight", 700)
      .attr("fill", variable("--texto")).text(u);
  });

  // red generada
  const red = d3.select("#agmRed").html("");
  agm.aristas.forEach(([u, v]) =>
    red.append("line").attr("x1", agm.pos[u][0]).attr("y1", agm.pos[u][1]).attr("x2", agm.pos[v][0])
      .attr("y2", agm.pos[v][1]).attr("stroke", variable("--arista")).attr("stroke-width", 2)
  );
  if (actual) {
    const [a, b] = [agm.pos[actual.u], agm.pos[actual.v]];
    red.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1])
      .attr("stroke", actual.agrega ? color(2) : color(1)).attr("stroke-width", 4)
      .attr("stroke-dasharray", actual.agrega ? null : "7 6");
  }
  agm.pos.forEach(([x, y], u) => {
    if (actual && (u === actual.u || u === actual.v))
      red.append("circle").attr("cx", x).attr("cy", y).attr("r", 22).attr("fill", variable("--resalte"));
    torta(red, x, y, 16, pesos[u]);
    red.append("text").attr("x", x).attr("y", y + 5).attr("text-anchor", "middle").attr("font-weight", 700)
      .attr("font-size", 13).attr("fill", "#fff").text(u);
  });
  red.append("text").attr("x", 10).attr("y", 322).attr("font-size", 12).attr("fill", variable("--suave"))
    .text(`pares revisados: ${agm.idx} de 45, aristas: ${agm.aristas.length}`);

  // explicación del paso
  const exp = $("agmExp");
  if (!actual) {
    exp.innerHTML = `<span class="etapa">Inicio</span><p>El grafo de afiliación ya está fijo: A = {0..4}, B = {3..7}, C = {7, 8, 9}.
      Pulse <b>Siguiente par</b> para tomar un par de nodos, calcular su probabilidad y lanzar la moneda.</p>`;
  } else {
    const nombres = actual.compartidas.map((c) => NOMBRES[c]);
    const producto = actual.compartidas.map((c) => `(1 − ${fmt(p[c])})`).join(" · ");
    exp.innerHTML = `<span class="etapa">Par ${agm.idx} de 45</span>
      <p>Par (${actual.u}, ${actual.v}). Comunidades compartidas C<sub>uv</sub> = {${nombres.join(", ") || "ninguna"}}.</p>
      <div class="formula">p(${actual.u}, ${actual.v}) = 1 − (1 − ε)${producto ? " · " + producto : ""} = ${fmt3(actual.prob)}</div>
      <p>Moneda: número aleatorio r = ${fmt3(actual.r)}. ${actual.agrega
        ? `<span class="ok">r &lt; p: se agrega la arista.</span>`
        : `<span class="no">r ≥ p: no hay arista.</span>`}</p>
      <p>${actual.compartidas.length > 1 ? "Compartir dos comunidades da dos oportunidades de conectarse: por eso el solape es más denso." : ""}</p>`;
  }
  agmBarras();
}

function agmBarras() {
  const svg = d3.select("#agmBarras").html("");
  const datos = [0, 1, 2].map((k) => {
    const h = agm.historial.filter((d) => d.compartidas.length === k);
    return { k, total: h.length, tasa: h.length ? d3.mean(h, (d) => +d.agrega) : 0 };
  });
  const x = d3.scaleBand().domain([0, 1, 2]).range([60, 500]).padding(0.35);
  const y = d3.scaleLinear().domain([0, 1]).range([180, 15]);
  ejes(svg, x, y, 180, 60, "comunidades compartidas por el par", "fracción con arista");
  svg.selectAll("rect.barra").data(datos).join("rect").attr("class", "barra")
    .attr("x", (d) => x(d.k)).attr("width", x.bandwidth()).attr("y", (d) => y(d.tasa))
    .attr("height", (d) => 180 - y(d.tasa)).attr("fill", (d) => (d.k === 0 ? variable("--suave") : color(d.k === 1 ? 0 : 2)));
  svg.selectAll("text.valor").data(datos).join("text").attr("class", "valor")
    .attr("x", (d) => x(d.k) + x.bandwidth() / 2).attr("y", (d) => y(d.tasa) - 6).attr("text-anchor", "middle")
    .attr("font-size", 12).attr("fill", variable("--texto")).text((d) => (d.total ? `${fmt(d.tasa)} (${d.total} pares)` : ""));
}

let agmTemporizador = null;
function agmReproducir() {
  if (agmTemporizador) {
    clearInterval(agmTemporizador);
    agmTemporizador = null;
    $("agmPlay").textContent = "Reproducir";
    return;
  }
  $("agmPlay").textContent = "Pausa";
  agmTemporizador = setInterval(() => {
    if (!agmPaso()) return agmReproducir();
    agmDibujar();
  }, 450);
}

// ==================================================================
// Pestaña 2: máxima verosimilitud
// ==================================================================
function mleDibujar() {
  const n = +$("mleN").value;
  $("mleK").max = n;
  const k = Math.min(+$("mleK").value, n);
  const p0 = +$("mleP").value;
  const usarLog = $("mleLog").checked;
  $("mleNv").textContent = n;
  $("mleKv").textContent = k;
  $("mlePv").textContent = fmt(p0);

  const logL = (p) => k * Math.log(p) + (n - k) * Math.log(1 - p);
  const f = (p) => (usarLog ? logL(p) : Math.exp(logL(p)));
  const ps = d3.range(0.005, 0.996, 0.005);
  const svg = d3.select("#mleMoneda").html("");
  const x = d3.scaleLinear().domain([0, 1]).range([60, 500]);
  const valores = ps.map(f);
  const y = d3.scaleLinear().domain(usarLog ? [Math.max(d3.min(valores), -60), d3.max(valores)] : [0, d3.max(valores)])
    .nice().range([250, 15]);
  ejes(svg, x, y, 250, 60, "p", usarLog ? "log L(p)" : "L(p)");
  svg.append("path").datum(ps).attr("fill", "none").attr("stroke", color(0)).attr("stroke-width", 3)
    .attr("d", d3.line().x((p) => x(p)).y((p) => y(Math.max(f(p), y.domain()[0]))));
  const pMax = Math.min(Math.max(k / n, 0.005), 0.995);
  svg.append("line").attr("x1", x(pMax)).attr("x2", x(pMax)).attr("y1", 250).attr("y2", y(f(pMax)))
    .attr("stroke", color(1)).attr("stroke-dasharray", "6 5").attr("stroke-width", 2);
  svg.append("circle").attr("cx", x(p0)).attr("cy", y(Math.max(f(p0), y.domain()[0]))).attr("r", 7).attr("fill", color(2));
  svg.append("text").attr("x", x(pMax)).attr("y", 12).attr("text-anchor", "middle").attr("fill", color(1))
    .attr("font-weight", 700).text("p̂ = " + fmt(k / n));

  // AGM: verosimilitud de la red generada al variar p_A
  const { p, eps } = agmParametros();
  const aristas = agmCompleto();
  const curva = d3.range(0.02, 0.99, 0.02).map((pa) => ({
    pa, l: AGM.logVerosimilitud(10, aristas, agm.membresias, [pa, p[1], p[2]], eps),
  }));
  const mejor = d3.greatest(curva, (d) => d.l);
  const svg2 = d3.select("#mleAGM").html("");
  const x2 = d3.scaleLinear().domain([0, 1]).range([60, 500]);
  const y2 = d3.scaleLinear().domain(d3.extent(curva, (d) => d.l)).nice().range([250, 15]);
  ejes(svg2, x2, y2, 250, 60, "p_A propuesto", "log L(G | B, p)");
  svg2.append("path").datum(curva).attr("fill", "none").attr("stroke", color(0)).attr("stroke-width", 3)
    .attr("d", d3.line().x((d) => x2(d.pa)).y((d) => y2(d.l)));
  svg2.append("line").attr("x1", x2(mejor.pa)).attr("x2", x2(mejor.pa)).attr("y1", 250).attr("y2", y2(mejor.l))
    .attr("stroke", color(1)).attr("stroke-dasharray", "6 5").attr("stroke-width", 2);
  svg2.append("line").attr("x1", x2(p[0])).attr("x2", x2(p[0])).attr("y1", 250).attr("y2", 15)
    .attr("stroke", color(2)).attr("stroke-width", 2);
  svg2.append("text").attr("x", x2(mejor.pa) + 6).attr("y", 30).attr("fill", color(1)).attr("font-weight", 700)
    .text("p̂_A = " + fmt(mejor.pa));
  svg2.append("text").attr("x", x2(p[0]) + 6).attr("y", 50).attr("fill", color(2)).attr("font-weight", 700)
    .text("p_A real = " + fmt(p[0]));

  $("mleExp").innerHTML = `<span class="etapa">Lectura</span>
    <p>Moneda: con ${k} caras en ${n} lanzamientos, L(p) es máxima en p̂ = ${k}/${n} = ${fmt(k / n)}.
    El punto verde es su propuesta: L(${fmt(p0)}) = ${usarLog ? fmt(logL(p0)) : Math.exp(logL(p0)).toExponential(2)}.
    El logaritmo convierte el producto en suma y no mueve el máximo.</p>
    <div class="formula">d log L / dp = k/p − (n − k)/(1 − p) = 0  ⟹  p̂ = k / n</div>
    <p>AGM: la red de la pestaña 1 tiene ${aristas.length} aristas. Con B fijo, la log verosimilitud es cóncava en
    x = −log(1 − p), y su máximo en p<sub>A</sub> es ${fmt(mejor.pa)}. Con solo 10 nodos la estimación es ruidosa;
    con más pares se acerca al valor real. Lo difícil del AGM no es este paso sino buscar B, que es combinatorio.</p>`;
}

// ==================================================================
// Pestaña 3: BigCLAM paso a paso
// ==================================================================
const bc = { historial: [] };
const SUBPASOS = ["1. elegir nodo", "2. gradiente", "3. búsqueda lineal", "4. proyectar y actualizar"];

function bcNuevo() {
  const K = +$("bcK").value;
  const semilla = +$("bcSemilla").value;
  const solape = +$("bcSolape").value;
  $("bcSolapev").textContent = solape.toFixed(1);
  const g = AGM.plantado(24, K, solape, 0.55, 0.02, semilla);
  bc.n = g.n;
  bc.K = K;
  bc.verdad = g.membresias;
  bc.aristas = g.aristas;
  bc.vecinos = d3.range(g.n).map(() => []);
  g.aristas.forEach(([u, v]) => (bc.vecinos[u].push(v), bc.vecinos[v].push(u)));
  bc.pos = distribuir(g.n, g.aristas, 440, 380);
  bc.delta = BigCLAM.umbral(g.n, g.aristas.length);
  bc.rng = generador(semilla);
  const ini = BigCLAM.inicializar(bc.vecinos, K, bc.rng);
  bc.F = ini.F;
  bc.semillas = ini.semillas;
  bc.S = bc.F.reduce((s, f) => sumaVec(s, f), ceros(K));
  bc.traza = [BigCLAM.logVerosimilitud(bc.F, bc.aristas)];
  bc.iter = 1;
  bc.orden = permutar(bc.n, bc.rng);
  bc.idx = 0;
  bc.sub = -1;
  bc.detalle = null;
  bc.convergido = false;
  bc.historial = [];
  bcDibujar();
}

function bcCopia() {
  return JSON.parse(JSON.stringify({
    F: bc.F, S: bc.S, traza: bc.traza, iter: bc.iter, orden: bc.orden, idx: bc.idx, sub: bc.sub,
    detalle: bc.detalle, convergido: bc.convergido,
  }));
}

function bcSubpaso() {
  if (bc.convergido) return false;
  bc.historial.push(bcCopia());
  if (bc.historial.length > 600) bc.historial.shift();
  bc.sub = (bc.sub + 1) % 4;
  const u = bc.orden[bc.idx];
  const Yvec = bc.vecinos[u].map((v) => bc.F[v]);
  const sumaVecinos = Yvec.reduce((s, y) => sumaVec(s, y), ceros(bc.K));
  if (bc.sub === 0) {
    bc.detalle = { u };
  } else if (bc.sub === 1) {
    const resto = restaVec(restaVec(bc.S, bc.F[u]), sumaVecinos);
    bc.detalle.resto = resto;
    bc.detalle.g = BigCLAM.gradienteFila(bc.F[u], Yvec, resto);
  } else if (bc.sub === 2) {
    bc.detalle.busqueda = BigCLAM.busquedaLineal(bc.F[u], bc.detalle.g, Yvec, bc.detalle.resto);
  } else {
    const viejo = bc.F[u];
    const nuevo = bc.detalle.busqueda.nuevo;
    bc.S = sumaVec(bc.S, restaVec(nuevo, viejo));
    bc.F[u] = nuevo;
    bc.detalle.viejo = viejo;
    bc.idx++;
    if (bc.idx === bc.n) {
      bc.traza.push(BigCLAM.logVerosimilitud(bc.F, bc.aristas));
      const [a, b] = bc.traza.slice(-2);
      bc.detalle.finIteracion = true;
      if (Math.abs(b - a) < 1e-5 * Math.abs(a) || bc.iter >= 200) bc.convergido = true;
      bc.iter++;
      bc.idx = 0;
      bc.orden = permutar(bc.n, bc.rng);
    }
  }
  return true;
}

function bcNodoCompleto() {
  do { if (!bcSubpaso()) return false; } while (bc.sub !== 3);
  return true;
}

function bcIteracion() {
  const it = bc.iter;
  while (bc.iter === it && !bc.convergido) bcNodoCompleto();
}

function bcAtras() {
  const previo = bc.historial.pop();
  if (previo) Object.assign(bc, previo);
  bcDibujar();
}

function bcDibujar() {
  const d = bc.detalle;
  const u = d ? d.u : null;
  const vecinos = new Set(u !== null ? bc.vecinos[u] : []);

  // progreso
  $("bcProgreso").innerHTML =
    `<span class="activo">iteración ${bc.convergido ? bc.iter - 1 : bc.iter}</span>` +
    `<span>nodo ${Math.min(bc.idx + (bc.sub === 3 ? 0 : 1), bc.n)} de ${bc.n}</span>` +
    SUBPASOS.map((s, i) => `<span class="${i === bc.sub ? "activo" : ""}">${s}</span>`).join("") +
    (bc.convergido ? `<span class="activo">convergió</span>` : "");

  // red
  const svg = d3.select("#bcRed").html("");
  bc.aristas.forEach(([a, b]) => {
    const incide = u !== null && (a === u || b === u);
    svg.append("line").attr("x1", bc.pos[a][0]).attr("y1", bc.pos[a][1]).attr("x2", bc.pos[b][0]).attr("y2", bc.pos[b][1])
      .attr("stroke", incide ? variable("--acento") : variable("--arista")).attr("stroke-width", incide ? 2.6 : 1.1);
  });
  bc.pos.forEach(([x, y], v) => {
    const miembro = bc.F[v].filter((f) => f >= bc.delta).length;
    if (v === u) svg.append("circle").attr("cx", x).attr("cy", y).attr("r", 18).attr("fill", variable("--resalte"));
    else if (miembro >= 2 && bc.convergido)
      svg.append("circle").attr("cx", x).attr("cy", y).attr("r", 15).attr("fill", "none")
        .attr("stroke", color(2)).attr("stroke-width", 3);
    const nodo = torta(svg, x, y, 11, bc.F[v]);
    if (u !== null && v !== u && !vecinos.has(v)) nodo.attr("opacity", 0.35);
    svg.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle").attr("font-size", 9)
      .attr("font-weight", 700).attr("fill", "#fff").attr("pointer-events", "none").text(v);
  });

  // matriz F
  const maximo = Math.max(1, d3.max(bc.F.flat()));
  let html = `<table class="matriz"><tr><th>nodo</th>${d3.range(bc.K).map((c) =>
    `<th style="color:${color(c)}">${NOMBRES[c]}</th>`).join("")}</tr>`;
  bc.F.forEach((fila, v) => {
    html += `<tr class="${v === u ? "actual" : ""}"><th>${v}</th>` + fila.map((f, c) => {
      const fondo = d3.color(color(c));
      fondo.opacity = Math.min(1, f / maximo) * 0.85;
      return `<td style="background:${fondo}; font-weight:${f >= bc.delta ? 700 : 400}">${fmt(f)}</td>`;
    }).join("") + "</tr>";
  });
  $("bcMatriz").innerHTML = html + `</table><p class="explicacion">negrita: F<sub>uc</sub> ≥ δ = ${fmt(bc.delta)}</p>`;

  // explicación
  const exp = $("bcExp");
  if (!d) {
    exp.innerHTML = `<span class="etapa">Inicialización</span>
      <p>Semillas: vecindarios localmente mínimos (por conductancia) de los nodos ${bc.semillas.join(", ")}.
      Para cada semilla k se pone F[N(u), k] = 1. Los nodos fuera de toda semilla empiezan en blanco.</p>
      <p>l(F) inicial = ${fmt(bc.traza[0])}. Pulse <b>Siguiente subpaso</b>.</p>`;
  } else if (bc.sub === 0) {
    exp.innerHTML = `<span class="etapa">Elegir nodo</span>
      <p>Se actualiza solo la fila F<sub>${u}</sub> = ${vec(bc.F[u])}; todas las demás quedan fijas, así el subproblema es cóncavo.</p>
      <p>Vecinos N(${u}) = {${bc.vecinos[u].join(", ")}} (resaltados). Los demás nodos son no vecinos.</p>`;
  } else if (bc.sub === 1) {
    exp.innerHTML = `<span class="etapa">Gradiente</span>
      <div class="formula">∇l(F<sub>u</sub>) = Σ<sub>v ∈ N(u)</sub> F<sub>v</sub> e<sup>−F<sub>u</sub>·F<sub>v</sub></sup> / (1 − e<sup>−F<sub>u</sub>·F<sub>v</sub></sup>) − Σ<sub>v ∉ N(u)</sub> F<sub>v</sub></div>
      <p>Truco del paper: Σ<sub>v ∉ N(u)</sub> F<sub>v</sub> = S − F<sub>u</sub> − Σ<sub>N(u)</sub> F<sub>v</sub> = ${vec(d.resto)},
      con S = ${vec(bc.S)} guardada en memoria. Costo O(grado de u).</p>
      <p>∇l(F<sub>${u}</sub>) = ${vec(d.g)} (recortado a ±${G_MAX}). Positivo: la membresía debe crecer.</p>`;
  } else if (bc.sub === 2) {
    const filas = d.busqueda.pruebas.slice(0, 6).map((p) =>
      `<tr><td>${p.t.toPrecision(3)}</td><td>${fmt(p.l)}</td><td>${fmt(p.umbral)}</td>
       <td class="${p.acepta ? "ok" : "no"}">${p.acepta ? "acepta" : "reduce t"}</td></tr>`).join("");
    exp.innerHTML = `<span class="etapa">Búsqueda lineal</span>
      <p>Se prueba F<sub>u</sub> + t ∇l con t = 1, 0.3, 0.09… hasta que l(F<sub>u</sub>) mejore al menos 0.05 ∇l·Δ (Armijo).
      Objetivo actual: ${fmt(d.busqueda.l0)}.</p>
      <table class="matriz"><tr><th>t</th><th>l nuevo</th><th>mínimo exigido</th><th></th></tr>${filas}</table>`;
  } else {
    exp.innerHTML = `<span class="etapa">Proyectar y actualizar</span>
      <p>F<sub>${u}</sub>: ${vec(d.viejo)} → ${vec(bc.F[u])}. Los valores negativos se llevan a 0 (proyección).</p>
      <p>S se corrige sumando la diferencia, sin recorrer los demás nodos.</p>
      ${d.finIteracion ? `<p><b>Fin de la iteración:</b> l(F) = ${fmt(bc.traza[bc.traza.length - 1])}.</p>` : ""}
      ${bc.convergido ? bcResumen() : ""}`;
  }
  bcCurva();
}

function bcResumen() {
  const detectadas = d3.range(bc.K).map((c) => d3.range(bc.n).filter((v) => bc.F[v][c] >= bc.delta));
  const f1 = (a, b) => { const s = a.filter((x) => b.includes(x)).length; return s ? (2 * s) / (a.length + b.length) : 0; };
  const lado = (A, B) => d3.mean(A, (a) => d3.max(B, (b) => f1(a, b)));
  const F1 = (lado(bc.verdad, detectadas) + lado(detectadas, bc.verdad)) / 2;
  return `<p class="ok">Convergió: la mejora relativa fue menor que 0.001 %.</p>
    <p>Comunidades (F ≥ δ): ${detectadas.map((c, i) => `${NOMBRES[i]} = {${c.join(", ")}}`).join("; ")}.</p>
    <p>F1 contra las comunidades plantadas: <b>${fmt(F1)}</b>. Anillo verde: nodos en dos o más comunidades.</p>`;
}

function bcCurva() {
  const svg = d3.select("#bcCurva").html("");
  const x = d3.scaleLinear().domain([0, Math.max(5, bc.traza.length - 1)]).range([60, 400]);
  const y = d3.scaleLinear().domain(d3.extent(bc.traza.concat([bc.traza[0] + 1]))).nice().range([160, 10]);
  ejes(svg, x, y, 160, 60, "iteración", "l(F)");
  svg.append("path").datum(bc.traza).attr("fill", "none").attr("stroke", color(0)).attr("stroke-width", 2.5)
    .attr("d", d3.line().x((d, i) => x(i)).y((d) => y(d)));
  svg.selectAll("circle").data(bc.traza).join("circle").attr("cx", (d, i) => x(i)).attr("cy", (d) => y(d))
    .attr("r", 3.5).attr("fill", color(0));
}

let bcTemporizador = null;
function bcReproducir() {
  if (bcTemporizador) {
    clearTimeout(bcTemporizador);
    bcTemporizador = null;
    $("bcPlay").textContent = "Reproducir";
    return;
  }
  $("bcPlay").textContent = "Pausa";
  const tic = () => {
    if (!bcSubpaso()) return bcReproducir();
    bcDibujar();
    bcTemporizador = setTimeout(tic, 1250 - +$("bcVel").value);
  };
  tic();
}

// ==================================================================
// Pestaña 4: AGM dirigido (CoDA)
// ==================================================================
const dg = {};

function dReiniciar(semilla) {
  const red = CoDA.ejemplo(semilla);
  Object.assign(dg, red);
  dg.K = 2;
  dg.salientes = d3.range(red.n).map(() => []);
  dg.entrantes = d3.range(red.n).map(() => []);
  red.aristas.forEach(([u, v]) => (dg.salientes[u].push(v), dg.entrantes[v].push(u)));
  const noDirigido = d3.range(red.n).map((u) => [...new Set(dg.salientes[u].concat(dg.entrantes[u]))]);
  dg.rng = generador(5);
  const { F } = BigCLAM.inicializar(noDirigido, dg.K, dg.rng);
  dg.F = F.map((f, u) => (dg.salientes[u].length ? f.slice() : ceros(dg.K)));
  dg.H = F.map((f, u) => (dg.entrantes[u].length ? f.slice() : ceros(dg.K)));
  dg.traza = [CoDA.logVerosimilitud(dg.F, dg.H, red.aristas)];
  dg.paso = 0;
  dg.pos = d3.range(red.n).map((u) => {
    if (u < 10) return [70 + (u % 2) * 55, 40 + u * 30];
    if (u < 14) return [300, 90 + (u - 10) * 55];
    const k = u - 14;
    return [400 + 70 * Math.cos((k / 8) * 2 * Math.PI), 260 + 70 * Math.sin((k / 8) * 2 * Math.PI)];
  });
  // BigCLAM sobre la versión sin dirección, para comparar
  dg.bigclam = BigCLAM.ajustar(noDirigido, dg.K, generador(5));
  dDibujar();
}

function dPaso() {
  if (dg.paso >= 60) return false;
  if (dg.paso % 2 === 0) CoDA.mediaIteracion(dg.F, dg.H, dg.salientes, dg.rng);
  else CoDA.mediaIteracion(dg.H, dg.F, dg.entrantes, dg.rng);
  dg.paso++;
  dg.traza.push(CoDA.logVerosimilitud(dg.F, dg.H, dg.aristas));
  return true;
}

function dDibujar() {
  const svg = d3.select("#dRed").html("");
  const defs = svg.append("defs");
  defs.append("marker").attr("id", "punta").attr("viewBox", "0 0 10 10").attr("refX", 19).attr("refY", 5)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", variable("--arista"));
  dg.aristas.forEach(([u, v]) =>
    svg.append("line").attr("x1", dg.pos[u][0]).attr("y1", dg.pos[u][1]).attr("x2", dg.pos[v][0]).attr("y2", dg.pos[v][1])
      .attr("stroke", variable("--arista")).attr("stroke-width", 1.1).attr("marker-end", "url(#punta)")
  );
  const delta = BigCLAM.umbral(dg.n, dg.aristas.length / 2);
  dg.pos.forEach(([x, y], u) => {
    torta(svg, x, y, 12, sumaVec(dg.F[u], dg.H[u]));
    svg.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle").attr("font-size", 9)
      .attr("font-weight", 700).attr("fill", "#fff").text(u);
  });
  [["fans", 95, 18], ["famosos", 300, 60], ["amigos", 400, 360]].forEach(([t, x, y]) =>
    svg.append("text").attr("x", x).attr("y", y).attr("text-anchor", "middle").attr("font-weight", 700)
      .attr("fill", variable("--suave")).text(t)
  );

  const comparar = $("dComparar").checked;
  const maximo = Math.max(1, d3.max(dg.F.flat().concat(dg.H.flat())));
  const celda = (f, c) => {
    const fondo = d3.color(color(c));
    fondo.opacity = Math.min(1, f / maximo) * 0.85;
    return `<td style="background:${fondo}; font-weight:${f >= delta ? 700 : 400}">${fmt(f)}</td>`;
  };
  let html = `<table class="matriz"><tr><th>nodo</th><th>grupo</th><th>F<sub>1</sub></th><th>F<sub>2</sub></th>
    <th>H<sub>1</sub></th><th>H<sub>2</sub></th>${comparar ? "<th>BigCLAM 1</th><th>BigCLAM 2</th>" : ""}</tr>`;
  d3.range(dg.n).forEach((u) => {
    html += `<tr><th>${u}</th><td>${dg.grupo(u)}</td>${dg.F[u].map(celda).join("")}${dg.H[u].map(celda).join("")}
      ${comparar ? dg.bigclam[u].map(celda).join("") : ""}</tr>`;
  });
  $("dMatrices").innerHTML = html + "</table>";

  const fase = dg.paso === 0 ? "inicio" : dg.paso % 2 ? "se actualizó F con H fijo" : "se actualizó H con F fijo";
  $("dExp").innerHTML = `<span class="etapa">Media iteración ${dg.paso}: ${fase}</span>
    <div class="formula">∇l(F<sub>u</sub>) = Σ<sub>v: u→v</sub> H<sub>v</sub> e<sup>−F<sub>u</sub>·H<sub>v</sub></sup> / (1 − e<sup>−F<sub>u</sub>·H<sub>v</sub></sup>) − Σ<sub>v: no u→v</sub> H<sub>v</sub></div>
    <p>Es el mismo subproblema de BigCLAM, pero F<sub>u</sub> se compara con la H de los nodos a los que u apunta,
    y H<sub>v</sub> con la F de los nodos que apuntan a v. l(F, H) = ${fmt(dg.traza[dg.traza.length - 1])}.</p>
    <p>Observe la columna de dos modos: los fans (0 a 9) tienen F alto y H bajo; los famosos (10 a 13), H alto y F bajo.
    ${comparar ? "BigCLAM, sin dirección, no puede separar estos roles: los fans no se conectan entre sí y no forman un grupo denso." : ""}</p>`;
}

let dTemporizador = null;
function dReproducir() {
  if (dTemporizador) {
    clearInterval(dTemporizador);
    dTemporizador = null;
    $("dPlay").textContent = "Reproducir";
    return;
  }
  $("dPlay").textContent = "Pausa";
  dTemporizador = setInterval(() => {
    if (!dPaso()) return dReproducir();
    dDibujar();
  }, 500);
}

// ==================================================================
// eventos
// ==================================================================
function redibujarTodo() {
  agmDibujar();
  mleDibujar();
  bcDibujar();
  dDibujar();
}

document.querySelectorAll(".pestana").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".pestana").forEach((x) => x.classList.toggle("activa", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("activo", p.id === b.dataset.panel));
    if (b.dataset.panel === "mle") mleDibujar();
  })
);

function temaActual() {
  const forzado = document.documentElement.dataset.theme;
  if (forzado) return forzado;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function etiquetaTema() {
  $("tema").textContent = temaActual() === "dark" ? "Tema claro" : "Tema oscuro";
}
$("tema").addEventListener("click", () => {
  document.documentElement.dataset.theme = temaActual() === "dark" ? "light" : "dark";
  try { localStorage.setItem("tema", document.documentElement.dataset.theme); } catch (e) { /* sin almacenamiento */ }
  etiquetaTema();
  redibujarTodo();
});
try {
  const guardado = localStorage.getItem("tema");
  if (guardado) document.documentElement.dataset.theme = guardado;
} catch (e) { /* sin almacenamiento */ }
etiquetaTema();

["pA", "pB", "pC", "eps"].forEach((id) =>
  $(id).addEventListener("input", () => {
    $(id + "v").textContent = fmt(+$(id).value);
    agmReiniciar();
  })
);
$("semillaAGM").addEventListener("change", agmReiniciar);
$("agmPaso").addEventListener("click", () => { agmPaso(); agmDibujar(); });
$("agmPlay").addEventListener("click", agmReproducir);
$("agmTodo").addEventListener("click", () => { while (agmPaso()); agmDibujar(); });
$("agmReset").addEventListener("click", agmReiniciar);

["mleN", "mleK", "mleP", "mleLog"].forEach((id) => $(id).addEventListener("input", mleDibujar));

$("bcSub").addEventListener("click", () => { bcSubpaso(); bcDibujar(); });
$("bcNodo").addEventListener("click", () => { bcNodoCompleto(); bcDibujar(); });
$("bcIter").addEventListener("click", () => { bcIteracion(); bcDibujar(); });
$("bcPlay").addEventListener("click", bcReproducir);
$("bcAtras").addEventListener("click", bcAtras);
$("bcReset").addEventListener("click", bcNuevo);
["bcK", "bcSolape", "bcSemilla"].forEach((id) => $(id).addEventListener("change", bcNuevo));
$("bcSolape").addEventListener("input", () => ($("bcSolapev").textContent = (+$("bcSolape").value).toFixed(1)));

$("dPaso").addEventListener("click", () => { dPaso(); dDibujar(); });
$("dPlay").addEventListener("click", dReproducir);
$("dReset").addEventListener("click", () => dReiniciar());
$("dComparar").addEventListener("change", dDibujar);

agmReiniciar();
mleDibujar();
bcNuevo();
dReiniciar();
