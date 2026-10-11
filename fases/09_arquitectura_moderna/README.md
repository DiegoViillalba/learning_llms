# Fase 09: arquitectura moderna (estilo Llama)

**Objetivo:** modernizar tu GPT-2 con las piezas que usan los LLMs actuales (RoPE, RMSNorm,
SwiGLU, GQA, sin biases), cada una con un interruptor en la config, y **medir** su efecto con
ablaciones controladas.
**Datos:** TinyStories (el mismo pipeline de la fase 07).

## Lecturas (en orden)

1. **Su et al. (2021), [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864).** §3.2–3.4.
   Complemento muy claro: EleutherAI, [*Rotary Embeddings: A Relative Revolution*](https://blog.eleuther.ai/rotary-embeddings/).
   *Extrae:* rotar `q` y `k` por pares de dimensiones hace que `q_m · k_n` dependa solo de `m − n`.
2. **Zhang & Sennrich (2019), [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467).**
   *Extrae:* qué quita respecto a LayerNorm (centrado y bias) y por qué casi no se pierde nada.
3. **Shazeer (2020), [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202).**
   *Extrae:* la FFN SwiGLU y cómo ajustar la dimensión oculta (≈ 2/3 · 4C) para igualar
   parámetros.
4. **Ainslie et al. (2023), [*GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245).**
   *Extrae:* el punto intermedio entre MHA y MQA (fase 08), con `n_kv_head` grupos.
5. **Touvron et al. (2023), [*LLaMA: Open and Efficient Foundation Language Models*](https://arxiv.org/abs/2302.13971).** §2.2–2.4.
   *Extrae:* la "receta" que combina todo lo anterior.
6. **Lilian Weng, [*The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/).**
   Mapa del zoológico de variantes, para consultar.

**Opcionales (frontera, para cuando todo lo anterior funcione):**
- Keller Jordan, [*Muon: An optimizer for hidden layers*](https://kellerjordan.github.io/posts/muon/), y el
  historial de récords de [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- Dehghani et al. (2023), [*Scaling Vision Transformers to 22B Parameters*](https://arxiv.org/abs/2302.05442):
  QK-norm para estabilidad.
- Karpathy, [nanochat](https://github.com/karpathy/nanochat): cómo se ve un pipeline completo
  y moderno (pre-entrenamiento, *mid-training*, SFT y chat) a escala pequeña.

## Qué implementar

Cada punto con un flag en `GPTConfig`, para poder apagarlo y prenderlo.

- [ ] **RoPE**, que reemplaza a `wpe`. Precalcula senos y cosenos y aplícalos a `q` y `k`.
- [ ] **RMSNorm**, que reemplaza a LayerNorm.
- [ ] **SwiGLU** en la FFN, con la dimensión oculta ajustada.
- [ ] **GQA**: `n_kv_head` ≤ `n_head`, repitiendo `k` y `v` por grupo.
- [ ] Sin biases en las capas lineales.
- [ ] Que tu KV cache de la fase 08 siga funcionando con RoPE (offset de posición).
- [ ] **Ablaciones:** mismos datos, mismo número de tokens, misma semilla y mismo número de
      parámetros (aprox.). Una tabla con el val loss de cada variante.

## Cómo verificar

- [ ] **Propiedad relativa de RoPE:** para vectores fijos `q`, `k`, el producto
      `rope(q, m) · rope(k, n)` es igual para todos los pares `(m, n)` con el mismo `m − n`.
- [ ] Tu RMSNorm = `torch.nn.RMSNorm` (`allclose`).
- [ ] GQA con `n_kv_head == n_head` = tu MHA (mismos pesos). Con `n_kv_head == 1` es MQA.
- [ ] Generación con cache = sin cache, ahora con RoPE.
- [ ] El conteo de parámetros de cada variante está dentro de ±2 % del baseline.

## Preguntas que debes poder responder

1. ¿Qué ventaja tiene una codificación posicional **relativa** frente a una absoluta aprendida?
2. ¿Cuánta memoria de KV cache ahorra GQA con `n_kv_head = n_head / 4`?
3. ¿Por qué una ablación justa necesita igualar parámetros **y** tokens?
4. ¿Cuál cambio te dio más mejora por unidad de cómputo? ¿Coincide con lo que reportan los papers?

## Bitácora (ablaciones)

| Variante | params | tokens | val loss | Δ vs baseline | Notas |
|---|---|---|---|---|---|
| baseline (GPT-2) | | | | — | |
| + RoPE | | | | | |
| + RMSNorm | | | | | |
| + SwiGLU | | | | | |
| + GQA | | | | | |
| todo junto | | | | | |
