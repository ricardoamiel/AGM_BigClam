# AGM, BigCLAM y AGM dirigido

Material de la semana 13 sobre detección de comunidades superpuestas con modelos generativos.

## Contenido

1. **papers**: índice de los artículos originales y de las alternativas (papers/LEEME.md); los PDF no se versionan.
2. **implementacion**: código en Python, legible y sin dependencias de grafos para los modelos.
   1. agm.py: grafo de afiliación, proceso generativo, log verosimilitud y ajuste con Metropolis Hastings.
   2. bigclam.py: subproblema de fila, semillas por conductancia, ascenso coordinado, umbral δ y elección de K.
   3. coda.py: AGM dirigido con membresías F (salida) y H (entrada).
   4. alternativas.py: Louvain, Leiden, propagación de etiquetas, CPM, SLPA y NOCD (con o sin atributos).
   5. ego_splitting.py: Ego-splitting (Epasto et al., 2017), solape a partir de particiones.
   6. bnmf.py: NMF bayesiana con ARD (Psorakis et al., 2011), K automático.
   7. metricas.py, datos.py y experimentos.py.
3. **resultados**: salidas de experimentos.py en JSON.
4. **figs**: figuras SVG y PDF, y capturas del visualizador.
5. **presentacion**: AGM_BigCLAM.pptx.
6. **notebook**: AGM_BigCLAM.ipynb, ya ejecutado.
7. **visualizador**: index.html con d3 v7; se abre directamente en el navegador. La pestaña 5 recorre Ego-splitting,
   la NMF bayesiana y el entrenamiento de NOCD (datos_nocd.js lo genera experimentos.py).
8. **datos**: redes ego de Facebook de SNAP con sus círculos sociales; no se versionan, se descargan así:
   mkdir -p datos/facebook, luego curl -L https://snap.stanford.edu/data/facebook.tar.gz y tar -xzf dentro de esa carpeta.

## Cómo regenerar todo

1. cd implementacion y python experimentos.py (unos cinco minutos).

## Decisiones de implementación

1. BigCLAM usa ε = 10⁻⁸ como el paper y recorta el gradiente a ±10: sin el recorte, una arista entre nodos sin
   comunidad común produce un gradiente cercano a 10⁸ y la búsqueda lineal nunca acepta un paso.
2. La condición de Armijo se evalúa con el paso efectivo después de proyectar sobre F ≥ 0.
3. Para elegir K se usa la validación con 20 % de pares reservados; el BIC del paper elige K = 1 en estos grafos.
4. El ajuste del AGM parte de las mismas semillas de BigCLAM; con un B aleatorio la cadena colapsa en una comunidad.
5. En el paper del AGM la aceptación aparece como max(1, L'/L); se implementa la regla correcta, min(1, L'/L).
