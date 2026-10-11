# Fase 10: fundamentos de JAX

**Objetivo:** cambiar el chip de PyTorch (objetos con estado y autograd en tensores) a JAX
(funciones puras transformables: `grad`, `jit`, `vmap`, pytrees y PRNG explícito),
reimplementando las fases 01–03 en JAX puro y verificándolas contra tu código en PyTorch.
**Hardware:** JAX corre en CPU en macOS, y para estos modelos alcanza.

## Lecturas (en orden)

1. **Frostig, Johnson & Leary (2018), [*Compiling machine learning programs via high-level tracing*](https://mlsys.org/Conferences/doc/2018/146.pdf)** (el paper original de JAX, 3 páginas).
   *Extrae:* la idea de trazar funciones de Python para obtener un programa que se compila con XLA.
2. **[JAX 101](https://docs.jax.dev/en/latest/101/index.html)**, completo y en orden:
   arrays, transformaciones (`grad`, `vmap`), pytrees, números pseudoaleatorios y
   computaciones con estado.
3. **Páginas clave de los docs:** [*Just-in-time compilation*](https://docs.jax.dev/en/latest/jit-compilation.html),
   [*Automatic differentiation*](https://docs.jax.dev/en/latest/automatic-differentiation.html) y
   [*Tracing*](https://docs.jax.dev/en/latest/tracing.html).
4. **[🔪 JAX – The Sharp Bits 🔪](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).**
   Lectura obligada: funciones puras, actualizaciones *in-place* con `.at[]`, control de flujo
   y PRNG.
5. **Baydin et al. (2018)** (de la fase 00), relectura de §3: ahora mapea `jax.jvp` al modo
   forward y `jax.vjp` / `jax.grad` al modo reverse.
6. **[Autodidax: JAX core from scratch](https://docs.jax.dev/en/latest/autodidax.html).**
   Es el "micrograd" de JAX: cómo funcionan los tracers y las transformaciones por dentro.
   Lee al menos la parte 1 (JVP y `vmap`).
7. **[Optax](https://optax.readthedocs.io/en/latest/)**: cómo se componen los optimizadores
   como transformaciones de gradientes (`optax.chain`).

**Opcional:** [Equinox](https://docs.kidger.site/equinox/), una alternativa a Flax con módulos
que son pytrees y un estilo más parecido a PyTorch.

## Qué implementar

Todo en **JAX puro** (`jax.numpy`, `jax.random`, `jax.grad`, `jax.jit`), sin Flax:

- [ ] Bigramas (fase 01): parámetros como pytree (`dict`), pérdida como función pura y SGD
      con `jax.tree.map`.
- [ ] MLP de Bengio (fase 02): `init(key, config) -> params`, `forward(params, x)` y
      `loss(params, x, y)`, con un `train_step` usando `jax.value_and_grad` y `jax.jit`.
- [ ] PRNG explícito: `jax.random.split` para la inicialización, el muestreo de batches y el
      dropout.
- [ ] `vmap` en algún punto útil, por ejemplo pérdida por ejemplo o gradientes por ejemplo.
- [ ] **Tu AdamW funcional:** `init(params) -> state` y `update(grads, state, params) -> (updates, state)`,
      con la misma interfaz que Optax.
- [ ] Inspecciona tu función de pérdida con `jax.make_jaxpr` y entiende el jaxpr.

## Cómo verificar

- [ ] **Paridad con PyTorch:** copia los pesos iniciales de tu MLP en PyTorch a JAX. Con los
      mismos batches, las pérdidas y los gradientes coinciden (`np.allclose`, `atol` ~1e-5 en
      fp32).
- [ ] Tu AdamW = `optax.adamw` tras N pasos.
- [ ] `jit` y sin `jit` dan el mismo resultado. Mide la aceleración y cuánto tarda la primera
      llamada (compilación).
- [ ] Con la misma semilla obtienes exactamente la misma corrida dos veces.

## Preguntas que debes poder responder

1. ¿Por qué JAX exige funciones puras? ¿Qué se rompe si tu función tiene efectos secundarios?
2. ¿Qué es un *tracer* y por qué un `if x > 0:` sobre un valor traza falla dentro de `jit`?
3. ¿Qué provoca una recompilación? (Pista: formas y argumentos estáticos).
4. ¿Por qué el PRNG de JAX es explícito y qué problema resuelve frente al estado global?
5. Compara `loss.backward()` + `optimizer.step()` de PyTorch con
   `grads = grad(loss)(params)` + `optax.update` de JAX. ¿Dónde vive el estado en cada caso?

## Bitácora

| Fecha | Experimento | Resultado | ¿Paridad con PyTorch? | Notas |
|---|---|---|---|---|
| | | | | |
