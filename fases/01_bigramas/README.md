# Fase 01: modelo de bigramas

**Objetivo:** construir tu primer modelo de lenguaje (a nivel caracter) de dos formas,
contando frecuencias y como red neuronal entrenada con gradiente, y ver que convergen a lo mismo.
**Datos:** `data/names.txt`.

## Lecturas (en orden)

1. **Jurafsky & Martin, *Speech and Language Processing* (3.ª ed.), cap. 3 "N-gram Language Models".**
   [libro](https://web.stanford.edu/~jurafsky/slp3/)
   *Extrae:* estimación por máxima verosimilitud, perplejidad, suavizado de Laplace (add-one).
2. **Karpathy, *makemore* parte 1 (bigramas).** [video](https://www.youtube.com/watch?v=PaCmpygFfXo)
   *Extrae:* el puente entre la tabla de conteos y la capa lineal + softmax.
3. **Tim Vieira, [*Exp-normalize trick*](https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).**
   *Extrae:* por qué restar el máximo antes de `exp` (softmax y log-sum-exp estables).

**Opcional:** Shannon (1948), [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf),
§1–3. Ahí aparecen por primera vez las aproximaciones al inglés con n-gramas.

## Qué implementar

- [ ] Vocabulario de caracteres (`stoi`, `itos`) con un token especial `.` de inicio/fin (V = 27).
- [ ] Matriz de conteos `V×V`, normalización por filas y muestreo de nombres con
      `torch.multinomial` y un `torch.Generator` con semilla fija.
- [ ] Pérdida: NLL promedio en todo el dataset. Suavizado sumando 1 a los conteos.
- [ ] Versión neuronal: one-hot `@ W` (`W` de `V×V`), softmax y NLL, entrenada con descenso
      de gradiente usando autograd de PyTorch.
- [ ] **Tu propia cross-entropy**: log-softmax estable vía log-sum-exp y luego NLL.
- [ ] Regularización L2 sobre `W` y su comparación con el suavizado.

## Cómo verificar

- [ ] Pérdida inicial con `W ≈ 0` = `ln(27) ≈ 3.30` (distribución uniforme).
- [ ] NLL por conteos ≈ **2.45**, y la red neuronal converge al mismo valor.
- [ ] Tu cross-entropy = `F.cross_entropy` (`torch.allclose`), incluso con logits enormes
      (×1000) sin producir `nan`/`inf`.
- [ ] Los nombres muestreados se ven "casi" nombres, y con la misma semilla salen igual
      por ambos métodos.

## Preguntas que debes poder responder

1. ¿Qué relación hay entre NLL promedio, cross-entropy y perplejidad?
2. ¿Por qué la regularización L2 sobre `W` equivale a suavizar los conteos?
3. ¿Por qué multiplicar un one-hot por `W` es solo indexar una fila? (Esto es `nn.Embedding`).
4. ¿Qué limita fundamentalmente a un modelo de bigramas?

## Bitácora

| Fecha | Método | NLL train | Notas |
|---|---|---|---|
| | conteos | | |
| | red neuronal | | |
