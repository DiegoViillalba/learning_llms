# Fase 08: inferencia y eficiencia

**Objetivo:** entender dónde se va el tiempo en un Transformer y hacer la generación rápida y
controlable: KV cache, estrategias de muestreo y el algoritmo de FlashAttention (entendido e
implementado en PyTorch puro, no como kernel).

## Lecturas (en orden)

1. **Horace He (2022), [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html).**
   *Extrae:* cuándo una operación está limitada por cómputo, por ancho de banda de memoria o
   por overhead.
2. **Kipply, [*Transformer Inference Arithmetic*](https://kipp.ly/transformer-inference-arithmetic/).**
   *Extrae:* qué es el KV cache, cuánta memoria ocupa y por qué el *decoding* está limitado por
   memoria y el *prefill* por cómputo.
3. **Milakov & Gimelshein (2018), [*Online normalizer calculation for softmax*](https://arxiv.org/abs/1805.02867).**
   *Extrae:* el softmax en una sola pasada, la pieza clave de FlashAttention.
4. **Dao et al. (2022), [*FlashAttention*](https://arxiv.org/abs/2205.14135).** §2–3, Algoritmo 1.
   *Extrae:* tiling, recálculo en el backward y por qué evita materializar la matriz `T×T`.
5. **Holtzman et al. (2019), [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751).**
   *Extrae:* por qué el muestreo greedy y el beam search degeneran, y qué es nucleus sampling
   (top-p).
6. **Shazeer (2019), [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150)** (MQA).
   *Extrae:* el problema del ancho de banda del KV cache. Es la motivación de GQA en la fase 09.
7. PyTorch docs: [`F.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
   y el [tutorial de `torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

**Opcional:** Leviathan et al. (2022), [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192).

## Qué implementar

- [ ] **KV cache** en tu GPT: fase de *prefill* (todo el prompt) y fase de *decode*
      (un token a la vez), con el offset de posiciones correcto.
- [ ] Benchmark de tokens/s con y sin cache, para varias longitudes.
- [ ] Muestreo: temperatura, top-k, top-p y (opcional) penalización por repetición.
- [ ] **Atención por bloques con softmax online** en PyTorch puro (el algoritmo de
      FlashAttention). No será más rápida en Python; el objetivo es entenderla.
- [ ] Perfilado de un paso de entrenamiento y uno de inferencia con `torch.profiler` o
      cronometrando cada componente.
- [ ] (Opcional) Speculative decoding usando tu modelo pequeño como *draft* y uno más grande
      como verificador.

## Cómo verificar

- [ ] Generación **greedy** con cache = sin cache (tokens idénticos).
- [ ] Tu atención por bloques = la atención estándar (`allclose`) para varios tamaños de bloque,
      incluyendo `T` que no sea múltiplo del bloque.
- [ ] `top_k = 1` equivale a greedy; `top_p = 1.0` y `temperature = 1.0` equivalen a muestrear
      de la distribución completa.
- [ ] La memoria del KV cache medida coincide con tu fórmula: `2 · L · B · T · C · bytes`.

## Preguntas que debes poder responder

1. ¿Por qué el decoding sin cache es O(T²) por token y con cache O(T)?
2. ¿Cuánta memoria ocupa el KV cache de GPT-2 124M con contexto 1024 y batch 8?
3. ¿Por qué FlashAttention es más rápida si hace **más** FLOPs (recálculo)?
4. ¿Cuándo preferirías top-p sobre top-k?

## Bitácora

| Fecha | Experimento | T | tokens/s sin cache | tokens/s con cache | Notas |
|---|---|---|---|---|---|
| | | | | | |
