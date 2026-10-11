# Fase 05: tu nanoGPT v1 (a nivel caracter)

**Objetivo:** ensamblar un GPT completo, decoder-only, con la arquitectura de GPT-2, y un
ciclo de entrenamiento limpio. Entrenarlo en Tiny Shakespeare en tu Mac.
**Datos:** `data/tinyshakespeare.txt` (V = 65).

## Lecturas (en orden)

1. **Radford et al. (2018), [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** (GPT-1). §3.1 y §4.1.
   *Extrae:* el objetivo de modelado de lenguaje y la arquitectura decoder-only.
2. **Radford et al. (2019), [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** (GPT-2). §2.3 "Model".
   *Extrae:* cada cambio respecto a GPT-1: LayerNorm al inicio de cada bloque, LayerNorm final,
   pesos residuales escalados por 1/√N al inicializar y contexto de 1024.
3. **Press & Wolf (2017), [*Using the Output Embedding to Improve Language Models*](https://arxiv.org/abs/1608.05859).**
   *Extrae:* *weight tying* entre `wte` y `lm_head`, y por qué funciona.
4. **Karpathy, [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY)**, segunda mitad.
5. Releer: [*A Recipe for Training Neural Networks*](http://karpathy.github.io/2019/04/25/recipe/).

**Opcional:** Karpathy (2015), [*The Unreasonable Effectiveness of Recurrent Neural Networks*](http://karpathy.github.io/2015/05/21/rnn-effectiveness/),
de donde viene Tiny Shakespeare. Sirve como contexto histórico.

## Qué implementar

Archivos sugeridos: `model.py`, `train.py`, `sample.py`.

- [ ] `GPTConfig` (dataclass): `block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`,
      `dropout`, `bias`.
- [ ] `GPT`: `wte`, `wpe`, `L` bloques, `ln_f` y `lm_head` atado a `wte`.
- [ ] Inicialización: `N(0, 0.02)` y proyecciones residuales con std `0.02 / √(2L)`.
- [ ] `forward(idx, targets=None) -> (logits, loss)`.
- [ ] `generate(idx, max_new_tokens, temperature, top_k)`, recortando el contexto a `block_size`.
- [ ] Ciclo de entrenamiento: `get_batch`, AdamW, `estimate_loss()` (con `model.eval()` y
      `torch.no_grad()`, promediando varios batches de train y val), logging, guardar y cargar
      checkpoint en `runs/`.
- [ ] **Conteo de parámetros a mano** (fórmula en función de V, T, C, L) comparado con
      `sum(p.numel() ...)`.
- [ ] Curvas de train y val loss con matplotlib.

## Cómo verificar

- [ ] Pérdida inicial ≈ `ln(65) ≈ 4.17`.
- [ ] Sobreajuste de un solo batch hasta pérdida ≈ 0.
- [ ] Tu conteo analítico de parámetros = el de PyTorch. Ojo: con *weight tying* los pesos
      compartidos se cuentan una vez.
- [ ] Números de referencia (nanoGPT, `shakespeare_char`):
  - Config pequeña (`L=4, nh=4, C=128, T=64, B=12`, 2000 iteraciones, sin dropout):
    val ≈ **1.88**.
  - Config grande (`L=6, nh=6, C=384, T=256, B=64, dropout=0.2`, ~10.7M params, 5000 iteraciones):
    val ≈ **1.47–1.48**. Pruébala en `mps` y mide el tiempo por iteración.
- [ ] Las muestras se ven como Shakespeare: nombres de personajes, saltos de línea y verso.

## Preguntas que debes poder responder

1. ¿Dónde están los parámetros? ¿Qué porcentaje está en embeddings, atención y MLP?
2. ¿Por qué el *weight tying* ahorra parámetros y además ayuda?
3. ¿Por qué escalar la inicialización de las proyecciones residuales por 1/√(2L)?
4. ¿Cuándo empieza a sobreajustar? ¿Qué efecto tiene el dropout aquí?
5. ¿Qué hacen `temperature` y `top_k` con la distribución?

## Bitácora

| Fecha | L | nh | C | T | dropout | params | iters | train | val | s/iter | Notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

**Diferencias con la referencia ([nanoGPT](https://github.com/karpathy/nanoGPT)):**

-
