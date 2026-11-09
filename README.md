# Detección de comunidades superpuestas: AGM, BigCLAM y propuestas recientes

Material de apoyo para la semana 13 del curso de Data Mining (Detección de comunidades, parte 2). Responde a la actividad de la sesión: investigar el modelo AGM (grafo de afiliación, proceso generativo, estimación de máxima verosimilitud), el modelo BigCLAM (relajación continua con la matriz F, log verosimilitud y gradiente ascendente coordinado), el AGM dirigido con membresías de salida y de entrada, y alternativas más recientes que cubren sus debilidades.

**Visualizador interactivo:** [https://ricardoamiel.github.io/AGM_BigClam/](https://ricardoamiel.github.io/AGM_BigClam/).

## Contenido

| Carpeta o archivo | Qué contiene |
| ----------------- | ------------ |
| visualizador | Herramienta con d3 v7 para recorrer cada algoritmo paso a paso; se publica en GitHub Pages. |
| implementacion | AGM, BigCLAM, CoDA, Ego-splitting, NMF bayesiana y alternativas en Python, junto con el script de experimentos. |
| presentacion/AGM_BigCLAM.pptx | 37 diapositivas con fondo blanco y figuras vectoriales. |
| notebook/AGM_BigCLAM.ipynb | Notebook ejecutado que explica y corre cada modelo. |
| resultados | Resultados de los experimentos en JSON. |
| figs | Figuras en SVG y PDF, y capturas del visualizador. |
| papers/LEEME.md | Índice de los 22 artículos de referencia. |

## Modelos implementados

1. **AGM** (Yang y Leskovec, ICDM 2012): proceso generativo y ajuste por máxima verosimilitud, alternando las probabilidades por comunidad con Metropolis Hastings sobre el grafo de afiliación.
2. **BigCLAM** (Yang y Leskovec, WSDM 2013): ascenso coordinado fila por fila con búsqueda lineal, proyección y el truco de la suma que deja cada iteración en tiempo lineal en el número de aristas.
3. **CoDA o AGM dirigido** (Yang, McAuley y Leskovec, WSDM 2014): membresías F de salida y H de entrada, que detectan comunidades de dos modos.
4. **Ego-splitting** (Epasto, Lattanzi y Paes Leme, KDD 2017): solape a partir de dos pasadas de un algoritmo de partición.
5. **NMF bayesiana** (Psorakis et al., Phys. Rev. E 2011): una prior por comunidad elige el número de comunidades.
6. **NOCD** (Shchur y Günnemann, 2019): una red neuronal de grafos con la verosimilitud de BigCLAM, con o sin atributos.
7. Referencias de comparación: Louvain, Leiden, propagación de etiquetas, CPM y SLPA.

## Resultados principales

1. En grafos sintéticos, al crecer el solape las particiones caen y BigCLAM y NOCD se mantienen sobre F1 0.8.
2. En las redes ego de Facebook de SNAP, NOCD gana en tres de cuatro redes y Ego-splitting en la 414, en unas centésimas de segundo.
3. CoDA separa a los fans que solo envían aristas de los famosos que solo las reciben; BigCLAM sin dirección no puede.
4. La NMF bayesiana elige el número de comunidades sin validación, pero con mucho solape crea una comunidad por cada zona solapada.

## Visualizador

Tiene cinco pestañas: proceso generativo del AGM par por par, máxima verosimilitud, BigCLAM en cuatro subpasos con retroceso, AGM dirigido comparado con BigCLAM, y propuestas recientes (Ego-splitting paso a paso, NMF bayesiana en vivo y el entrenamiento de NOCD por época). Tiene tema claro y oscuro y se adapta a pantallas de teléfono. Funciona también abriendo visualizador/index.html directamente en el navegador, con conexión a internet para cargar d3.

### Publicación en GitHub Pages

GitHub Pages no detecta la carpeta visualizador por sí solo: con la opción de publicar desde una rama solo admite la raíz o la carpeta docs. Por eso el repositorio incluye el flujo .github/workflows/pages.yml, que publica esa carpeta. Pasos:

1. Subir el repositorio a GitHub con la rama main.
2. En el repositorio, abrir Settings, luego Pages, y en Build and deployment elegir Source: GitHub Actions.
3. Ir a la pestaña Actions y ejecutar el flujo Publicar visualizador en GitHub Pages con Run workflow (solo la primera vez; después se ejecuta solo en cada push que modifique el visualizador).
4. Al terminar, la dirección aparece en Settings, Pages, y en el resumen del flujo.

En la cuenta gratuita, GitHub Pages requiere que el repositorio sea público.

## Cómo reproducir

1. Instalar numpy, scipy, networkx, matplotlib, torch, igraph y leidenalg.
2. Descargar los datos de Facebook de SNAP en datos/facebook (instrucciones en LEEME.md).
3. Desde implementacion, ejecutar python experimentos.py; tarda unos cinco minutos y escribe resultados y visualizador/datos_nocd.js.
4. Desde notebook, ejecutar jupyter nbconvert --to notebook --execute --inplace AGM_BigCLAM.ipynb.

Los PDF de los papers y los datos de SNAP no se versionan; los primeros tienen derechos de autor y los segundos se descargan de su fuente.
