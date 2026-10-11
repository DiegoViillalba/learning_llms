# Fase 11: tu GPT en JAX

**Objetivo:** portar tu GPT (fases 05, 07 y opcionalmente 09) a JAX, primero en JAX puro y
después, si quieres, en Flax NNX. Debe ser numéricamente equivalente a tu versión en PyTorch.
**Datos:** Tiny Shakespeare (caracteres) para entrenar en CPU; pesos de GPT-2 para verificar.

## Lecturas (en orden)

1. **[Flax NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html)** y docs de Flax:
   `nnx.Module`, `nnx.Rngs`, `nnx.Optimizer` y `nnx.jit`. Solo si haces la versión en Flax.
2. **[Optax](https://optax.readthedocs.io/en/latest/):** `optax.adamw` (con `mask` para el
   weight decay), `optax.clip_by_global_norm`, `optax.warmup_cosine_decay_schedule` y
   `optax.chain`.
3. **Rabe & Staats (2021), [*Self-attention Does Not Need O(n²) Memory*](https://arxiv.org/abs/2112.05682).**
   *Extrae:* atención eficiente en memoria implementada **en JAX**. Es la contraparte de
   FlashAttention de la fase 08.
4. **Austin et al. (Google DeepMind, 2025), [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/).**
   Capítulos de rooflines, matemáticas del Transformer y entrenamiento. Es el puente hacia
   entrenar en TPUs/GPUs con JAX a escala.
5. **[*Distributed arrays and automatic parallelization*](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)**
   (opcional): sharding, para cuando tengas acceso a varios dispositivos.
6. **[*Train a miniGPT language model with JAX*](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html)**
   (JAX AI Stack). **Solo después de tener el tuyo funcionando**, para comparar.

## Qué implementar

**Versión 1, JAX puro:**
- [ ] `init_params(key, config)` que devuelva un pytree con la misma estructura de pesos que tu
      GPT de PyTorch.
- [ ] `forward(params, idx, *, train, key)` como función pura: embeddings, bloques (pre-LN,
      atención causal, MLP) y `ln_f` con *weight tying*.
- [ ] Atención causal con máscara, verificada contra `jax.nn.dot_product_attention`.
- [ ] `train_step` jiteado con Optax (clipping + AdamW con máscara de decay + schedule) y
      `donate_argnums` para reutilizar buffers.
- [ ] Dropout con PRNG explícito.
- [ ] Data loader con numpy (el mismo formato de shards de la fase 07).
- [ ] Checkpoints con `np.savez` (o Orbax, opcional).
- [ ] `generate` con KV cache de **tamaño fijo** (formas estáticas) y un bucle con
      `jax.lax.fori_loop` o un bucle de Python sobre un paso jiteado.
- [ ] (Avanzado) Apilar los pesos de las `L` capas y recorrerlas con `jax.lax.scan`.
      Compara el tiempo de compilación.

**Versión 2, Flax NNX (opcional):** el mismo modelo con `nnx.Module`, para comparar ergonomía.

## Cómo verificar

- [ ] **La prueba principal:** carga en tu GPT de JAX los pesos de tu GPT de PyTorch (o los de
      GPT-2 de OpenAI). Los logits coinciden (`np.allclose`, `atol` ~1e-4 en fp32).
- [ ] Generación greedy con pesos de GPT-2 = la de tu versión en PyTorch.
- [ ] La curva de pérdida en Tiny Shakespeare es comparable a la de la fase 05 con la misma
      config.
- [ ] Cache = sin cache en generación greedy.
- [ ] No hay recompilaciones inesperadas después del primer paso. Detéctalas con
      `jax.log_compiles` o midiendo tiempos.

## Preguntas que debes poder responder

1. ¿Por qué el KV cache en JAX tiene que tener tamaño fijo y cómo manejas la posición actual?
2. ¿Cuánto tarda la compilación y cómo la reduce `lax.scan` sobre las capas?
3. ¿Qué hace `donate_argnums` y por qué importa la memoria?
4. Después de implementar lo mismo en ambos frameworks: ¿qué te gusta de cada uno y qué te
   cuesta?
5. ¿Qué cambiarías en tu código para entrenarlo en una TPU v4-8 con data parallelism?

## Bitácora

| Fecha | Experimento | ¿Paridad logits? | s/iter (CPU) | val loss | Notas |
|---|---|---|---|---|---|
| | | | | | |
