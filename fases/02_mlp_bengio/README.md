# Fase 02: MLP con embeddings (Bengio et al. 2003)

**Objetivo:** pasar de un caracter de contexto a `block_size` caracteres usando embeddings
aprendidos y una capa oculta, y aprender el oficio básico del entrenamiento: splits, minibatches
y búsqueda de learning rate.
**Datos:** `data/names.txt`.

## Lecturas (en orden)

1. **Bengio, Ducharme, Vincent & Jauvin (2003), [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), JMLR.**
   Lee a fondo las §1–3 y la Figura 1.
   *Extrae:* la matriz de embeddings compartida `C`, la concatenación del contexto y la
   arquitectura exacta (ecuación de `y`). Es el ancestro directo de GPT.
2. **Karpathy, *makemore* parte 2 (MLP).** [video](https://www.youtube.com/watch?v=TCH_1BHY58I)
3. **Karpathy (2019), [*A Recipe for Training Neural Networks*](http://karpathy.github.io/2019/04/25/recipe/).**
   Léelo ahora y reléelo en cada fase.

**Opcional:** Mikolov et al. (2013), [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781)
(word2vec), para intuición sobre qué capturan los embeddings.

## Qué implementar

- [ ] Dataset con ventana de contexto (`block_size = 3`) y split train/val/test 80/10/10 con
      semilla fija.
- [ ] Tabla de embeddings `C` (`V × d`), concatenación, capa oculta `tanh` y capa de salida.
      Escribe las shapes en cada línea.
- [ ] Entrenamiento con minibatches aleatorios.
- [ ] **Búsqueda de learning rate:** barrido exponencial de lr (p. ej. 1e-3 → 1) durante unas
      pocas iteraciones, graficando la pérdida contra log(lr).
- [ ] Decaimiento del lr en escalón al final del entrenamiento.
- [ ] Muestreo de nombres nuevos.
- [ ] Gráfica 2D de los embeddings (con `d = 2`, o con PCA si `d > 2`).

## Cómo verificar

- [ ] **Sobreajuste de un minibatch** (p. ej. 32 ejemplos): la pérdida debe bajar a ~0.
- [ ] Pérdida inicial cercana a `ln(27) ≈ 3.30`. Si es mucho mayor, tu inicialización produce
      logits demasiado confiados (lo arreglas en la fase 03).
- [ ] val loss ≈ **2.2** o menos.
- [ ] En la gráfica de embeddings, las vocales quedan agrupadas.

## Preguntas que debes poder responder

1. ¿Por qué compartir la matriz `C` entre todas las posiciones del contexto?
2. ¿Cómo crece el número de parámetros con `block_size`? ¿Por qué eso no escala? (Motivación
   para la atención).
3. ¿Qué indica una brecha grande entre train y val loss? ¿Y que sean casi iguales?
4. ¿Cómo eliges el learning rate a partir del barrido?

## Bitácora

| Fecha | block_size | d (emb) | oculta | params | train loss | val loss | Notas |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
