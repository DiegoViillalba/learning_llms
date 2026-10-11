# Fase 00: autograd escalar (micrograd)

**Objetivo:** entender backpropagation implementando un motor de diferenciación automática en
modo reverso sobre escalares, y entrenar un MLP pequeño con él.

## Lecturas (en orden)

1. **Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*.**
   [video](https://www.youtube.com/watch?v=VMj-3S1tku0)
   *Extrae:* qué es un grafo computacional, la regla de la cadena local en cada nodo y por qué
   el recorrido hacia atrás va en orden topológico inverso.
2. **Karpathy (2016), [*Yes you should understand backprop*](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b).**
   *Extrae:* qué falla cuando tratas backprop como caja negra (saturación, neuronas muertas,
   gradientes que explotan).
3. **Parr & Howard (2018), [*The Matrix Calculus You Need For Deep Learning*](https://arxiv.org/abs/1802.01528).**
   Secciones de jacobianos y regla de la cadena vectorial.
   *Extrae:* el paso de escalares a vectores y matrices, que vas a necesitar desde la fase 01.
4. **Baydin et al. (2018), [*Automatic Differentiation in Machine Learning: a Survey*](https://arxiv.org/abs/1502.05767).**
   Secciones 2–3.
   *Extrae:* diferencia entre modo **forward** (JVP) y **reverse** (VJP). Te servirá de nuevo
   en JAX (`jax.jvp`, `jax.vjp`, `jax.grad`).
5. PyTorch docs, [*Autograd mechanics*](https://docs.pytorch.org/docs/stable/notes/autograd.html).

**Opcional:** 3Blue1Brown, [capítulos 3–4 de redes neuronales](https://www.3blue1brown.com/topics/neural-networks)
(intuición visual de backprop).

## Qué implementar

- [ ] Clase `Value` que guarde dato, gradiente, hijos y una función `_backward` local.
- [ ] Operaciones: `+`, `*`, `**` (exponente constante), `-`, `/`, `exp`, `tanh`, `relu`, y
      las versiones reflejadas (`__radd__`, `__rmul__`, …).
- [ ] `backward()`: orden topológico y recorrido inverso.
- [ ] `Neuron`, `Layer`, `MLP` con método `parameters()`.
- [ ] Ciclo de entrenamiento: forward, pérdida, `zero_grad`, backward, paso de SGD manual.
- [ ] Entrenar en un dataset 2D de juguete (p. ej. dos lunas generadas a mano con numpy).
- [ ] (Extra) Visualizar el grafo con graphviz.

## Cómo verificar

- [ ] **Gradiente numérico:** compara cada gradiente con diferencias finitas centradas
      `(f(x+h) − f(x−h)) / 2h`. El error relativo debe ser < 1e-6.
- [ ] **Contra PyTorch:** construye la misma expresión con tensores `float64` y
      `requires_grad=True`. Los gradientes deben coincidir.
- [ ] **Nodo reutilizado:** `b = a + a` debe dar `a.grad == 2`. Si da 1, estás asignando en
      vez de acumular.
- [ ] El MLP logra ~100 % de exactitud en train en el dataset de juguete.

## Preguntas que debes poder responder

1. ¿Por qué los gradientes se **acumulan** (`+=`) en vez de asignarse?
2. ¿Por qué hace falta el orden topológico?
3. ¿Por qué en ML se usa modo reverso y no forward? ¿Cuándo convendría el forward?
4. ¿Qué tiene que ver `zero_grad()` con la pregunta 1?

## Bitácora

| Fecha | Experimento | Resultado | Notas |
|---|---|---|---|
| | | | |

**Diferencias con la referencia ([micrograd](https://github.com/karpathy/micrograd)):**

-
