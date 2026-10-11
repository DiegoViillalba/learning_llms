# Fase 04: atención y el bloque Transformer

**Objetivo:** implementar self-attention causal multi-cabeza y el bloque Transformer
(pre-LN + atención + MLP + residuales), entendiendo cada línea.
**Datos:** `data/tinyshakespeare.txt` a nivel caracter (V = 65).

## Lecturas (en orden)

1. **Bahdanau, Cho & Bengio (2014), [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473).** §3.
   *Extrae:* la atención como promedio ponderado aprendido. Es el origen de la idea.
2. **Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/).**
   Intuición visual antes del paper.
3. **Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).**
   Céntrate en §3.1–3.5. GPT es solo el *decoder* sin cross-attention, así que ignora el
   encoder y los detalles de traducción.
   *Extrae:* ecuación (1) y **demuestra por qué se escala por 1/√d_k** (varianza del producto
   punto), multi-head (§3.2.2), FFN (§3.3) y codificación posicional (§3.5).
4. **Phuong & Hutter (2022), [*Formal Algorithms for Transformers*](https://arxiv.org/abs/2207.09238).**
   Pseudocódigo preciso de cada pieza. Tu referencia principal para implementar **sin mirar
   código**. Concéntrate en los algoritmos de atención, `MHAttention`, `layer_norm`,
   `DTransformer` (decoder-only) y `DTraining`/`DInference`.
5. **Xiong et al. (2020), [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745).**
   *Extrae:* por qué pre-LN (la que usa GPT-2) entrena más estable que el post-LN del paper
   original y no necesita warmup tan cuidadoso.
6. **Hendrycks & Gimpel (2016), [*GELU*](https://arxiv.org/abs/1606.08415).**
7. **Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY).**
   Primera mitad, hasta tener un bloque completo. Incluye el "truco matemático" de la máscara
   triangular.
8. **Elhage et al. (2021, Anthropic), [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html).**
   Lee la parte del *residual stream* y de las cabezas de atención como operaciones que leen
   y escriben en él. Cambia tu forma de pensar el modelo.

**Opcionales:**
- Lilian Weng, [*Attention? Attention!*](https://lilianweng.github.io/posts/2018-06-24-attention/), panorama histórico.
- 3Blue1Brown, [capítulos de Transformers y atención](https://www.3blue1brown.com/topics/neural-networks).
- Harvard NLP, [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/).
  **Tiene código: léelo después de tu implementación.**

## Qué implementar

- [ ] *Baseline*: modelo de bigramas en Tiny Shakespeare (`nn.Embedding(V, V)`), con
      `get_batch` de trozos aleatorios de longitud `T` y split 90/10.
- [ ] "Truco matemático" de tres formas, comprobando que dan lo mismo: promedio de tokens
      previos con un bucle, con `matmul` por una matriz triangular normalizada y con
      `softmax` sobre una máscara de `-inf`.
- [ ] Una cabeza de self-attention causal: proyecciones `q, k, v`, escala, máscara y softmax.
- [ ] Multi-head, primero como lista de cabezas concatenadas y luego en versión **batched**
      con `view`/`transpose` a `(B, nh, T, hs)`, más la proyección de salida.
- [ ] FFN: `Linear(C, 4C)` → GELU → `Linear(4C, C)`.
- [ ] `Block` pre-LN: `x = x + attn(ln1(x))`, `x = x + mlp(ln2(x))`.
- [ ] Embeddings posicionales aprendidos.

## Cómo verificar

- [ ] Tu atención = `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (`allclose`).
- [ ] **Causalidad:** si cambias los tokens en posiciones > t, las salidas en ≤ t no cambian.
      Equivalente: el gradiente de la salida t respecto a entradas futuras es 0.
- [ ] Multi-head batched = lista de cabezas, copiando los mismos pesos.
- [ ] **Permutación:** sin máscara ni embeddings posicionales, permutar la entrada solo
      permuta la salida. Esto demuestra por qué hacen falta las posiciones.
- [ ] **Escala:** con `q, k ~ N(0, 1)`, la varianza de `q·k` es ≈ `hs`, y tras dividir
      entre √hs es ≈ 1. Mídelo.

## Preguntas que debes poder responder

1. ¿Qué pasa con el softmax si no escalas por 1/√d_k y `hs` es grande?
2. ¿Por qué la máscara usa `-inf` antes del softmax y no ceros después?
3. ¿Qué es el *residual stream* y qué leen y escriben en él las capas?
4. ¿Por qué la atención por sí sola es un operador sobre *conjuntos*?
5. ¿Pre-LN o post-LN? Explica la diferencia en el flujo del gradiente.

## Bitácora

| Fecha | Modelo | params | train loss | val loss | Notas |
|---|---|---|---|---|---|
| | bigramas | | | | |
| | 1 bloque | | | | |
