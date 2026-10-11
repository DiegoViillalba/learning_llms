# Fase 07: GPT-2 (124M) y un ciclo de entrenamiento serio

**Objetivo:** escalar tu GPT a la arquitectura exacta de GPT-2 124M, demostrar que es correcta
cargando los pesos de OpenAI, y construir un ciclo de entrenamiento con todas las piezas
"de verdad": AdamW con grupos de decay, warmup + cosine, gradient clipping, gradient
accumulation, precisión mixta, evaluación y checkpoints.
**Datos:** TinyStories tokenizado (local). FineWeb-Edu solo si alquilas GPU.

## Lecturas (en orden)

1. **GPT-2 paper, completo** ([PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
2. **Brown et al. (2020), [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165)** (GPT-3). §2.1–2.3, **Tabla 2.1** y **Apéndice B**.
   *Extrae:* hiperparámetros por tamaño de modelo (lr 6e-4 y batch de 0.5M tokens para 125M),
   AdamW con β₁ = 0.9, β₂ = 0.95, ε = 1e-8, clipping a 1.0, weight decay de 0.1, warmup lineal
   y decaimiento cosine. GPT-2 no publicó estos detalles; por eso se toman de GPT-3.
3. **Karpathy, [*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU).**
   El repo [build-nanogpt](https://github.com/karpathy/build-nanogpt) tiene un commit por paso,
   así que úsalo como solucionario incremental y consulta solo el commit del paso en el que estés.
4. **Loshchilov & Hutter (2016), [*SGDR*](https://arxiv.org/abs/1608.03983)**, para el schedule cosine.
5. **Goyal et al. (2017), [*Accurate, Large Minibatch SGD*](https://arxiv.org/abs/1706.02677).** §2.2, warmup.
6. **Pascanu, Mikolov & Bengio (2013), [*On the difficulty of training recurrent neural networks*](https://arxiv.org/abs/1211.5063).** §3.2, gradient clipping.
7. **Micikevicius et al. (2017), [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740)**, junto con los
   [docs de AMP de PyTorch](https://docs.pytorch.org/docs/stable/amp.html).
8. **Kaplan et al. (2020), [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361).** §1–2 y Tabla 1.
   *Extrae:* `C ≈ 6·N·D` (FLOPs de entrenamiento ≈ 6 × parámetros × tokens).
9. **Hoffmann et al. (2022), [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556)** (Chinchilla).
   *Extrae:* la regla de ~20 tokens por parámetro.
10. **EleutherAI, [*Transformer Math 101*](https://blog.eleuther.ai/transformer-math/).**
    Memoria (pesos, gradientes, estados de Adam, activaciones) y FLOPs.
11. **Chowdhery et al. (2022), [*PaLM*](https://arxiv.org/abs/2204.02311), Apéndice B**, con la
    definición de MFU (*Model FLOPs Utilization*).
12. **Zellers et al. (2019), [*HellaSwag*](https://arxiv.org/abs/1905.07830).** Formato de la tarea.
13. **Eldan & Li (2023), [*TinyStories*](https://arxiv.org/abs/2305.07759).**
    *Extrae:* por qué un modelo diminuto puede producir texto coherente con datos simples.

## Qué implementar

- [ ] `GPT.from_pretrained("gpt2")`: carga los pesos de Hugging Face (`transformers`, grupo
      `ref`) en **tu** modelo. Ojo: HF usa capas `Conv1D` cuyos pesos están **transpuestos**
      respecto a `nn.Linear`.
- [ ] Generación con esos pesos y tu tokenizador o `tiktoken`.
- [ ] Script de preparación de datos: tokenizar TinyStories y guardar *shards* de `np.uint16`
      en `data/`. Un `DataLoader` que los lea con `np.memmap`.
- [ ] `configure_optimizers`: weight decay solo en tensores de 2 o más dimensiones (matmuls y
      embeddings), sin decay en biases ni LayerNorm.
- [ ] Schedule del lr: warmup lineal y luego cosine hasta un 10 % del máximo.
- [ ] Gradient clipping por norma global (1.0); registra la norma en cada paso.
- [ ] **Gradient accumulation** para simular batches grandes. Cuidado con la escala de la pérdida.
- [ ] Precisión mixta con `torch.autocast` (bf16; en `mps` compruébalo, y si falla usa fp32).
- [ ] `F.scaled_dot_product_attention` en lugar de tu atención manual. `torch.compile`, si
      funciona en tu backend.
- [ ] Vocabulario rellenado a 50304 (múltiplo de 64) y medición de su efecto en la velocidad.
- [ ] Medición de tokens/s y **MFU**.
- [ ] Evaluación de HellaSwag: para cada ejemplo, pérdida promedio en los tokens de cada uno
      de los 4 finales y elegir el menor.
- [ ] Checkpoints con estado del optimizador y del paso, para poder reanudar.
- [ ] **Entrenamiento local:** modelo de ~10–30M parámetros en TinyStories con tu tokenizador
      de la fase 06 (vocab de 4096 a 8192, para que los embeddings no dominen).

## Cómo verificar

- [ ] **La prueba principal:** con los pesos de OpenAI, tus logits = los de
      `GPT2LMHeadModel` de `transformers` en la misma entrada (máx. diferencia < 1e-4 en fp32).
- [ ] Pérdida inicial con pesos aleatorios ≈ `ln(50257) ≈ 10.82`.
- [ ] GPT-2 124M pre-entrenado en HellaSwag (validación): exactitud ≈ **29–30 %**.
- [ ] Gradient accumulation: k micro-batches acumulados dan **los mismos gradientes** que un
      batch k veces más grande.
- [ ] Al reanudar desde un checkpoint, la curva de pérdida continúa sin saltos.
- [ ] **Ejercicio de presupuesto:**
  1. Mide los TFLOPS reales de tu Mac con un benchmark de matmul en `mps`.
  2. Mide tus tokens/s entrenando.
  3. Con `C ≈ 6ND`, estima cuánto tardaría entrenar 124M parámetros con 10B tokens.
  4. Decide con números qué tamaño de modelo y cuántos tokens sí caben en una noche.

## Preguntas que debes poder responder

1. ¿Por qué hace falta warmup? ¿Qué pasa con Adam en los primeros pasos sin él?
2. ¿Por qué β₂ = 0.95 y no 0.999 en LLMs?
3. ¿Por qué no aplicar weight decay a biases ni a LayerNorm?
4. ¿Qué es MFU, cuánto obtienes y dónde se pierde el resto?
5. ¿Cuánta memoria necesita entrenar 124M con AdamW en fp32? Desglósala.
6. Según Chinchilla, ¿cuántos tokens "merece" tu modelo local?

## Bitácora

| Fecha | Modelo | params | tokens vistos | tokens/s | MFU | val loss | HellaSwag | Notas |
|---|---|---|---|---|---|---|---|---|
| | GPT-2 (pesos OpenAI) | 124M | — | — | — | | | |
| | | | | | | | | |
