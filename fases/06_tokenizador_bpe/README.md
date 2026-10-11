# Fase 06: tokenizador BPE a nivel byte

**Objetivo:** implementar desde cero el tokenizador Byte Pair Encoding a nivel byte que usa
GPT-2/GPT-4 (entrenar, codificar y decodificar) y reproducir exactamente a `tiktoken`.
**Datos:** Tiny Shakespeare para pruebas y una muestra de TinyStories (algunos MB) para
entrenar el tokenizador que usarás en la fase 07.

## Lecturas (en orden)

1. **Nathan Reed, [*A Programmer's Introduction to Unicode*](https://www.reedbeta.com/blog/programmers-intro-to-unicode/).**
   *Extrae:* code points y UTF-8 (de 1 a 4 bytes por caracter). Sin esto el BPE a nivel byte
   no se entiende.
2. **Sennrich, Haddow & Birch (2016), [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909).**
   §3.2 y el Algoritmo 1.
   *Extrae:* el bucle de BPE (contar pares, fusionar el más frecuente, repetir).
3. **GPT-2 paper, §2.2 "Input Representation"** ([PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
   *Extrae:* BPE sobre bytes (vocabulario base de 256, nunca hay `<UNK>`) y por qué impiden
   fusiones entre categorías de caracteres (la pre-tokenización con regex).
4. **Karpathy, [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE).**
5. **[minbpe/exercise.md](https://github.com/karpathy/minbpe/blob/master/exercise.md).**
   Sigue sus pasos como enunciado, **sin abrir el código del repo**.
6. [README de tiktoken](https://github.com/openai/tiktoken): cómo usarlo como referencia para
   tus pruebas.

**Opcionales:** Kudo & Richardson (2018), [*SentencePiece*](https://arxiv.org/abs/1808.06226).
Kudo (2018), [*Subword Regularization*](https://arxiv.org/abs/1804.10959) (tokenizador unigram).

## Qué implementar

- [ ] `BasicTokenizer`: `train(text, vocab_size)`, `encode(text)` y `decode(ids)` sobre bytes
      UTF-8.
- [ ] `RegexTokenizer`: pre-tokenización con el patrón regex de GPT-2/GPT-4 (necesita el
      paquete `regex`, que soporta `\p{L}`, ya instalado) y BPE dentro de cada trozo.
- [ ] Tokens especiales (`<|endoftext|>`) con su manejo en `encode`.
- [ ] `save`/`load` de merges y vocabulario.
- [ ] Reproducir `tiktoken` con la codificación `cl100k_base` (GPT-4), cargando sus rangos en
      tu tokenizador (paso 4 del `exercise.md`). Pista: `tiktoken` solo expone
      `mergeable_ranks` y tienes que **reconstruir los merges**.
- [ ] Entrenar tu tokenizador en una muestra de TinyStories con `vocab_size` de 4096 u 8192.
      Lo usarás en la fase 07.

## Cómo verificar

- [ ] `decode(encode(s)) == s` para texto arbitrario: español con acentos, emojis, código y
      espacios múltiples.
- [ ] Tus ids = `tiktoken.get_encoding("cl100k_base").encode(s)` en un corpus de prueba variado.
- [ ] Reporta la razón de compresión (bytes por token) de tu tokenizador contra la de GPT-2 en
      el mismo texto.
- [ ] Con `vocab_size = 256` + 0 merges, el tokenizador equivale a usar los bytes crudos.

## Preguntas que debes poder responder

1. ¿Por qué los LLMs fallan al deletrear, contar letras o invertir cadenas?
2. ¿Por qué el BPE a nivel byte no necesita token `<UNK>`?
3. ¿Qué compensaciones hay entre el tamaño del vocabulario, la longitud de secuencia y el
   tamaño de la matriz de embeddings?
4. ¿Qué problema resuelve la pre-tokenización con regex?
5. ¿Por qué el texto en español "cuesta" más tokens que el inglés con el tokenizador de GPT-2?

## Bitácora

| Fecha | Corpus | vocab_size | bytes/token | Tiempo de entrenamiento | Notas |
|---|---|---|---|---|---|
| | | | | | |
