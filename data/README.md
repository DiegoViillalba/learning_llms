# Datos

Todo lo que hay en esta carpeta (salvo este archivo) lo ignora git. Corre los comandos desde
la raíz del repo.

## Pequeños (descárgalos ya)

**names.txt**: 32 mil nombres, uno por línea. Fases 01–03.

```bash
curl -L -o data/names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

**Tiny Shakespeare**: ~1 MB de texto. Fases 04, 05, 10 y 11.

```bash
curl -L -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**HellaSwag (validación)**: evaluación de la fase 07. El repo original en GitHub
(`rowanz/hellaswag`) ya no está disponible, y el script de `build-nanogpt` que lo descarga de
ahí va a fallar. Usa la copia de Hugging Face,
[`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) (split `validation`, ~10k
ejemplos), con la librería `datasets` desde tu script de evaluación.

## Medianos (fase 07 en adelante)

**TinyStories** ([Eldan & Li, 2023](https://arxiv.org/abs/2305.07759);
[dataset](https://huggingface.co/datasets/roneneldan/TinyStories)): cuentos cortos con
vocabulario simple. Con él, modelos de pocos millones de parámetros ya generan texto coherente,
así que es ideal para entrenar en tu Mac. Descárgalo con la librería `datasets` (grupo `ref`)
desde tu propio script de preparación. El script debe tokenizar y guardar *shards* de
`uint16` en `data/tinystories/`.

## Grandes (solo si alquilas GPU)

**FineWeb-Edu**, subconjunto `sample-10BT`
([dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)): ~10B tokens, el que usa
Karpathy para reproducir GPT-2 124M. No lo descargues en local.
