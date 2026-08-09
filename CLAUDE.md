# Interpretabilidad de ViT — penalización de similitud entre cabezas de atención

## Contexto de la tesis

Título: *Interpretabilidad de redes neuronales a partir de penalización de representaciones complejas de conceptos*.

Hilo del trabajo: penalizar información compartida/redundancia entre componentes de una red para inducir representaciones más localizadas y por tanto más interpretables.

- **Capítulo ya cerrado (CNN):** penalización de información mutua entre filtros de la última capa convolucional (asignando filtros a categorías). Resultado: filtros más localizados y menos solapados, con una caída pequeña de desempeño vs. el modelo sin penalizar.
- **Capítulo actual (ViT):** se extiende el principio a cabezas de atención. Se identificó y documentó que un **producto punto entre matrices de cabezas de atención** puede usarse como penalización de similitud entre cabezas (análogo del término de información compartida en CNN, pero aplicado a Q/K/V o a las salidas de cada cabeza). Esta parte ya está escrita en `main.tex` (secciones "Información compartida en modelos de atención" en adelante).
- **Siguiente capítulo (este):** no es proponer una penalización nueva, es **interpretar y analizar la estructura interna del modelo a partir de las diferencias que produce esa penalización** — comparar modelo penalizado vs. baseline y extraer evidencia de que las cabezas quedaron menos redundantes y más especializadas.

## Código y datos disponibles (`experimentation/`)

- `utils/model.py`: ViT propio (`ViTForClassfication`), con `output_attentions=True` devolviendo `attention_probs` por bloque, shape `(B, H, T, T)`.
- `utils/visualizations.py`: `visualize_attention` (grid de heatmaps por cabeza) y `visualize_attention_on_images` (overlay de atención sobre la imagen, por cabeza y bloque).
- `utils/experiments.py`: guardar/cargar checkpoints + config + métricas (`experiments/<nombre>/`).
- `training.ipynb`, `notebooks/visualize_attention.ipynb`.
- Datasets ya montados: CIFAR-10, MNIST, Oxford-IIIT Pet, **CelebA** (con `list_attr_celeba.csv` = atributos por imagen y `list_bbox_celeba.csv` = bounding box de la cara). CelebA es el más valioso para este capítulo porque permite validar cuantitativamente contra etiquetas semánticas y contra localización espacial real, no solo comparar cualitativamente.

## Qué falta para este capítulo

Necesitas al menos dos modelos entrenados y guardados (mismo config, misma seed de datos): uno **baseline** (sin penalización) y uno **penalizado** (con el término de producto punto entre cabezas). Todo el análisis de abajo es comparativo entre ambos. Se organiza en tres bloques, siguiendo el marco estándar de interpretabilidad mecanística (feature / circuit / universality, cf. Olah et al., *Zoom In: An Introduction to Circuits*).

### A. Feature study — ¿qué representa cada cabeza y qué tan distintas son?
1. **Similitud + rango efectivo** (geométrico): matriz de similitud coseno/producto punto por pares de cabezas por capa (`(L,H,H)`), resumen escalar por capa, y rango efectivo (Roy & Vetterli 2007) del espectro de valores singulares de cada cabeza — evidencia cuantitativa directa de separación en el espacio de representación. Repetir sobre Q/K/V, sobre la salida de la cabeza, y sobre los mapas de atención (¿penalizar pesos vs. comportamiento da resultados distintos?). CKA (Kornblith et al. 2019) como métrica secundaria, más robusta, para no razonar en círculo con la misma forma funcional de la penalización.
2. **Logit-lens en el CLS**: proyectar el CLS token intermedio de cada bloque (post-LN) a través del clasificador final para rastrear en qué capa "se decide" la clase. Barato con tu arquitectura porque el clasificador ya es lineal sobre el CLS final.
3. **Probing por cabeza**: clasificador lineal pequeño sobre el output de cada cabeza para predecir una propiedad conocida (atributos de `list_attr_celeba.csv`, o pseudo-etiquetas de color/textura por parche si se necesita algo más granular). Etiqueta semánticamente qué aprendió cada cabeza, no solo que son distintas matemáticamente.

### B. Circuit study — ¿cómo interactúan las cabezas y cuáles importan causalmente?
4. **Ablación de nodo**: apagar una cabeza a la vez y medir caída de accuracy (Michel et al. 2019 "Are Sixteen Heads Really Better than One?"; Voita et al. 2019). Hipótesis: baseline tiene cabezas prescindibles y otras críticas; penalizado reparte la importancia más uniformemente. Incluye contribución directa al logit por cabeza.
5. **Path patching**: ¿cambiaron las relaciones *entre* cabezas tras penalizar, no solo cada cabeza por separado? Para un ViT de clasificación hay que definir la "corrupted run" (p.ej. imagen de otra clase) y parchear la activación de una cabeza dentro de la corrida limpia, midiendo el efecto en otra cabeza o en el logit. Es la pieza más costosa del plan — si el tiempo aprieta, sustituto barato: correlación entre patrones de atención de pares de cabezas capa a capa (no causal, pero rápido de calcular y ya sugiere si vale la pena ir por path patching completo).

### C. Universality study — ¿es un hallazgo real o le tocó a esa corrida?
6. **2-3 semillas de entrenamiento**, mismo modelo y penalización, distinta inicialización. No repetir todo el análisis por semilla — correr las semillas solo sobre las métricas núcleo (similitud/rango efectivo + ablación), y dejar probing/logit-lens/path patching sobre una semilla representativa.

### Complemento cualitativo (transversal, barato)
- `visualize_attention_on_images` sobre el mismo set fijo de imágenes, baseline vs. penalizado, lado a lado — elegir 3-4 ejemplos para la tesis (ya existe la figura de bloques 3-4 en `main.tex`, se amplía a comparación).
- Entropía de cada mapa de atención por cabeza/bloque como proxy numérico de dispersión.

## Cómo priorizar

Orden de mayor a menor payoff/esfuerzo: **(A.1) similitud+rango efectivo → (B.4) ablación de cabezas → (C.6) universality sobre A.1+B.4 → (A.2) logit-lens → (A.3) probing → (B.5) path patching** (o su sustituto barato de correlación entre cabezas, si no alcanza el tiempo para path patching completo). A.1 y B.4 sostienen por sí solas el argumento central ("la penalización reduce redundancia funcional, no solo geométrica"); C.6 es lo que blinda el capítulo ante la pregunta obligada del jurado; A.3 conecta con el marco de "conceptos" del resto de la tesis.

## Referencias clave por bloque

- **Feature study**: Kornblith, Norouzi, Lee, Hinton (2019), *Similarity of Neural Network Representations Revisited* (CKA) — [proceedings.mlr.press/v97/kornblith19a](http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf). Roy & Vetterli (2007), *The Effective Rank: A Measure of Effective Dimensionality*. nostalgebraist (2020), *interpreting GPT: the logit lens* (blog, origen del método; adaptaciones a ViT existen, ver Medium "Unlocking Visual Insights: Applying the Logit Lens to Image Data with Vision Transformers"). Probing clásico: Alain & Bengio (2016), *Understanding intermediate layers using linear classifier probes*.
- **Circuit study**: Michel, Levy, Neubig (2019), *Are Sixteen Heads Really Better than One?* — [papers.neurips.cc/paper/9551](http://papers.neurips.cc/paper/9551-are-sixteen-heads-really-better-than-one.pdf). Voita, Talbot, Moiseev, Sennrich, Titov (2019), *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned* — [arxiv.org/abs/1905.09418](https://arxiv.org/abs/1905.09418). Path patching: Wang et al. (2022), *Interpretability in the Wild* (circuito IOI en GPT-2, referencia metodológica aunque sea en NLP).
- **Universality study**: Olah et al. (2020), *Zoom In: An Introduction to Circuits* (Distill) — marco general de universalidad de features/circuitos entre corridas.

## Convenciones de trabajo

- Comparaciones siempre baseline vs. penalizado, mismo dataset, misma arquitectura/config, mismo checkpoint final (o misma época) — nunca comparar entre datasets distintos sin dejarlo explícito.
- Guardar cada experimento nuevo con `save_experiment` en `experiments/<nombre-descriptivo>/` para que quede trazable en la tesis.
- Terminología del documento: seguir usando "penalización de información compartida" / "redundancia" / "especialización" tal como aparece en `main.tex`, no introducir sinónimos nuevos sin necesidad.
