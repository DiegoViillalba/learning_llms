# Fase 03: entrenar redes profundas (init, normalización, optimizadores, backprop manual)

**Objetivo:** entender por qué las redes profundas son difíciles de entrenar y qué piezas lo
resuelven: inicialización, normalización, residuales y Adam/AdamW. Todas reaparecen en el
Transformer. También aprenderás la API de `nn.Module`.
**Datos:** `data/names.txt` (sigue el MLP de la fase 02, pero más profundo).

## Lecturas (en orden)

1. **Glorot & Bengio (2010), [*Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html).**
   *Extrae:* el argumento de preservar la varianza de activaciones y gradientes entre capas.
2. **He et al. (2015), [*Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852).** §2.2.
   *Extrae:* la inicialización Kaiming y el factor de ganancia según la no linealidad.
3. **Ioffe & Szegedy (2015), [*Batch Normalization*](https://arxiv.org/abs/1502.03167).** Algoritmos 1 y 2.
   *Extrae:* la normalización en entrenamiento contra la de inferencia (estadísticas móviles).
4. **Ba, Kiros & Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450).**
   *Extrae:* sobre qué eje se normaliza y por qué no depende del batch. **Es la que usa GPT.**
5. **He et al. (2016), [*Deep Residual Learning*](https://arxiv.org/abs/1512.03385).** §3.1–3.2.
   *Extrae:* la conexión residual como "autopista" para el gradiente.
6. **Kingma & Ba (2014), [*Adam*](https://arxiv.org/abs/1412.6980).** Algoritmo 1 y §3 (corrección de sesgo).
7. **Loshchilov & Hutter (2017), [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101)** (AdamW).
   *Extrae:* por qué en Adam el weight decay no equivale a una penalización L2, y el Algoritmo 2.
8. **Srivastava et al. (2014), [*Dropout*](https://jmlr.org/papers/v15/srivastava14a.html).** §2 y §4.
9. **Karpathy, *makemore* partes 3, 4 y 5.**
   [3: activaciones, gradientes y BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) ·
   [4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) ·
   [5: WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

**Opcional:** van den Oord et al. (2016), [*WaveNet*](https://arxiv.org/abs/1609.03499), base de *makemore* 5.

## Qué implementar

- [ ] MLP de 5 o más capas con `tanh`. Con mala inicialización, grafica histogramas de
      activaciones y gradientes por capa y muestra la saturación.
- [ ] Inicialización Kaiming manual. Repite las gráficas.
- [ ] Ratio `update / data` por parámetro (en log10, debería rondar −3).
- [ ] **Tu BatchNorm1d y tu LayerNorm** (forward, con modos train/eval en BN).
- [ ] **Backprop manual** del MLP completo con BN, sin `loss.backward()` (como el ejercicio de
      *makemore* 4): cross-entropy, capa lineal, BN, `tanh` y embeddings.
- [ ] **Tu optimizador AdamW** (clase con `step()` y `zero_grad()`).
- [ ] Dropout "invertido" propio.
- [ ] Reorganiza todo como subclases de `nn.Module` (`Linear`, `LayerNorm`, `MLP`) usando
      `nn.Parameter`, `model.train()` y `model.eval()`.

## Cómo verificar

- [ ] Cada gradiente manual = el de autograd (máx. diferencia absoluta ~1e-9 en `float64`).
- [ ] Tu LayerNorm = `nn.LayerNorm` y tu BatchNorm = `nn.BatchNorm1d`, en train y en eval.
- [ ] **Tu AdamW = `torch.optim.AdamW`:** mismos pesos iniciales y mismos gradientes, N pasos,
      pesos idénticos (`allclose`).
- [ ] Con buena inicialización la pérdida inicial es ≈ `ln(27)`, sin la "curva de palo de
      hockey" del principio.
- [ ] El dropout escala bien: en eval la salida es determinista y el valor esperado no cambia.

## Preguntas que debes poder responder

1. ¿Por qué BatchNorm es mala idea en un modelo autoregresivo y LayerNorm no?
2. ¿Qué corrige la corrección de sesgo de Adam en los primeros pasos?
3. ¿Por qué en Adam el weight decay desacoplado ≠ L2?
4. ¿Cómo ayuda una conexión residual al flujo del gradiente? Escribe la derivada.
5. ¿Qué te dice el ratio update/data y qué harías si es 1e-1? ¿Y si es 1e-6?

## Bitácora

| Fecha | Config | train loss | val loss | Notas |
|---|---|---|---|---|
| | | | | |
