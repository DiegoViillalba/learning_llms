# Learning LLMs: un GPT desde cero (PyTorch → JAX)

Ruta de estudio para implementar **tú mismo, desde cero y en local**, un modelo de lenguaje
pequeño estilo [nanoGPT](https://github.com/karpathy/nanoGPT) en PyTorch y después portarlo a JAX.

Aquí no hay código. Hay papers, artículos y videos en orden, y para cada fase: qué implementar,
cómo saber que quedó bien y qué preguntas deberías poder responder al terminar.

---

## 1. Método de trabajo

Cada fase sigue el mismo ciclo:

1. **Leer** las lecturas en el orden indicado. Para los papers usa el método de tres pasadas
   de Keshav ([*How to Read a Paper*](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf), 2007):
   1. Vistazo (10 min): título, abstract, figuras, conclusiones.
   2. Comprensión: ecuaciones y figuras, saltándote las pruebas.
   3. Reconstrucción: ¿podrías reimplementarlo con solo tus notas?
   Usa la plantilla [notas/_plantilla_paper.md](notas/_plantilla_paper.md).
2. **Derivar en papel**: escribe las formas (*shapes*) de cada tensor y las ecuaciones con tu
   notación (ver §4) antes de abrir el editor.
3. **Implementar** sin mirar código de referencia.
4. **Verificar** con las pruebas de la fase: contra PyTorch, gradientes numéricos y números de
   referencia conocidos. Escribe esas pruebas como `test_*.py` dentro de la carpeta de la fase
   y córrelas con `uv run pytest fases/0X_.../`.
5. **Recién entonces**, compara con la implementación de referencia (§6) y anota en la bitácora
   de la fase qué hiciste distinto y por qué.

**Sobre los videos de Karpathy:** en ellos escribe el código en pantalla, así que verlos
equivale a ver la solución. Úsalos como clase teórica: velo completo **sin teclear**, toma notas
conceptuales, ciérralo e implementa desde tus notas. Vuelve al video solo si llevas más de ~45
minutos atorado, y mira únicamente el fragmento que necesitas.

**Regla de oro (de [*A Recipe for Training Neural Networks*](http://karpathy.github.io/2019/04/25/recipe/)):**
antes de entrenar en serio, verifica que la pérdida inicial sea la esperada y que tu modelo
pueda sobreajustar (*overfit*) un solo batch hasta pérdida ≈ 0. Si no puede, hay un bug.

---

## 2. Entorno

Ya está creado con [uv](https://docs.astral.sh/uv/) (Python 3.12):

| Grupo | Paquetes | Para qué |
|---|---|---|
| base | `torch`, `numpy`, `matplotlib`, `tqdm` | todo el trabajo en PyTorch |
| `dev` | `jupyterlab`, `ipykernel`, `pytest` | experimentar y probar |
| `ref` | `tiktoken`, `transformers`, `datasets` | **solo para verificar** tu código (comparar tokens, cargar pesos de GPT-2, descargar datasets) |
| `jax` | `jax`, `flax`, `optax` | fases 10–11 |

Comandos útiles (`train.py` es un ejemplo: ese archivo lo escribirás tú):

```bash
uv sync
```

```bash
uv run python fases/05_gpt_caracteres/train.py
```

```bash
uv run jupyter lab
```

```bash
uv run pytest fases/
```

**Notas para tu Mac (Apple M5, 24 GB):**
- PyTorch usa la GPU con el backend `mps`: `torch.device("mps")` si
  `torch.backends.mps.is_available()`. Si alguna operación no está soportada, exporta
  `PYTORCH_ENABLE_MPS_FALLBACK=1`. Ver [notas de MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
- `torch.compile` y la precisión mixta en MPS son menos maduros que en CUDA. Si algo falla,
  entrena en fp32 sin compile.
- JAX corre en **CPU** en macOS. Alcanza para las fases 10–11 con modelos pequeños. Si quieres
  acelerador para JAX, usa Google Colab o Kaggle (TPU/GPU gratis).
- Entrenar GPT-2 124M con 10B tokens **no es viable en local** (lo calcularás en la fase 07).
  En local entrenarás modelos de ~1–30M de parámetros. Para la corrida grande, alquila una GPU
  por unas horas (opcional).

---

## 3. Estructura

```
learning_llms/
├── README.md                 ← esta guía (mapa general)
├── pyproject.toml / uv.lock  ← entorno
├── fases/                    ← tu código, una carpeta por fase
│   ├── 00_micrograd/README.md          (cada README: lecturas, tareas, verificación, bitácora)
│   ├── 01_bigramas/
│   ├── 02_mlp_bengio/
│   ├── 03_entrenar_redes_profundas/
│   ├── 04_atencion/
│   ├── 05_gpt_caracteres/
│   ├── 06_tokenizador_bpe/
│   ├── 07_gpt2_124m/
│   ├── 08_inferencia_eficiencia/
│   ├── 09_arquitectura_moderna/
│   ├── 10_jax_fundamentos/
│   └── 11_gpt_jax/
├── notas/                    ← una nota por paper (usa la plantilla)
├── papers/                   ← PDFs (ignorados por git)
├── data/                     ← datasets (ignorados por git; ver data/README.md)
└── runs/                     ← logs y checkpoints (ignorados por git)
```

Cada fase es autocontenida. Si una fase necesita algo de la anterior (p. ej. tu tokenizador en
la fase 07), cópialo a la nueva carpeta. Mejor aún: reescríbelo de memoria, que también es
práctica.

---

## 4. Convenciones de notación

Usa siempre la misma notación en notas y código (es la de nanoGPT):

| Símbolo | Significado |
|---|---|
| `B` | tamaño del batch |
| `T` | longitud de la secuencia (contexto, `block_size`) |
| `C` | dimensión del modelo (`n_embd`, *d_model*) |
| `V` | tamaño del vocabulario |
| `nh`, `hs` | número de cabezas y tamaño de cabeza (`C = nh · hs`) |
| `L` | número de capas (`n_layer`) |

Anota la forma esperada al lado de cada operación, por ejemplo `# (B, nh, T, hs)`.

---

## 5. Mapa de ruta

| Fase | Tema | Lecturas clave | Hito verificable |
|---|---|---|---|
| [00](fases/00_micrograd/) | Autograd escalar | Karpathy *micrograd*; Baydin et al. 2018 (AD survey) | Tus gradientes = gradiente numérico = PyTorch |
| [01](fases/01_bigramas/) | Modelo de bigramas | Jurafsky & Martin cap. 3 (n-gramas); *makemore* 1 | NLL ≈ 2.45 en `names.txt` por conteos **y** por red neuronal |
| [02](fases/02_mlp_bengio/) | MLP con embeddings | **Bengio et al. 2003**; *makemore* 2 | val loss ≈ 2.2 |
| [03](fases/03_entrenar_redes_profundas/) | Init, normalización, optimizadores | Glorot 2010, He 2015, BatchNorm, **LayerNorm**, ResNet, **Adam**, **AdamW** | Backprop manual = autograd; tu AdamW = `torch.optim.AdamW` |
| [04](fases/04_atencion/) | Atención y bloque Transformer | Bahdanau 2014; **Vaswani et al. 2017**; **Phuong & Hutter 2022**; Xiong 2020 (pre-LN) | Tu atención = `F.scaled_dot_product_attention`; prueba de causalidad |
| [05](fases/05_gpt_caracteres/) | **Tu nanoGPT v1** (caracteres) | **GPT-1**, **GPT-2** (§2.3), Press & Wolf 2017 | val loss < 1.9 en Tiny Shakespeare (≈ 1.5 con el modelo grande) |
| [06](fases/06_tokenizador_bpe/) | Tokenizador BPE | **Sennrich et al. 2016**; GPT-2 §2.2; Karpathy *minbpe* | Tus ids = `tiktoken` |
| [07](fases/07_gpt2_124m/) | GPT-2 124M y entrenamiento serio | **GPT-3** (Tabla 2.1, Ap. B), Kaplan 2020, Chinchilla, mixed precision, HellaSwag | Tus logits con pesos de OpenAI = `transformers` |
| [08](fases/08_inferencia_eficiencia/) | KV cache, *sampling*, FlashAttention | Kipply; **FlashAttention**; Holtzman 2019 | Generación con cache = sin cache |
| [09](fases/09_arquitectura_moderna/) | Estilo Llama | **RoPE**, RMSNorm, SwiGLU, **LLaMA**, GQA | Tabla de ablaciones con el mismo presupuesto |
| [10](fases/10_jax_fundamentos/) | Fundamentos de JAX | JAX 101, *Sharp Bits*, paper de JAX 2018, Autodidax | Bigramas/MLP en JAX = PyTorch |
| [11](fases/11_gpt_jax/) | **Tu GPT en JAX** | Flax NNX/Optax; Rabe & Staats 2021; *How to Scale Your Model* | Logits JAX = logits PyTorch con los mismos pesos |

El núcleo son las fases 04 a 07. Las fases 00–03 puedes acelerarlas si ya dominas backprop y
el entrenamiento de MLPs, pero no te saltes sus pruebas de verificación.

### Progreso

- [ ] 00 micrograd
- [ ] 01 bigramas
- [ ] 02 MLP de Bengio
- [ ] 03 Entrenar redes profundas
- [ ] 04 Atención
- [ ] 05 GPT de caracteres
- [ ] 06 Tokenizador BPE
- [ ] 07 GPT-2 124M
- [ ] 08 Inferencia y eficiencia
- [ ] 09 Arquitectura moderna
- [ ] 10 Fundamentos de JAX
- [ ] 11 GPT en JAX

---

## 6. Implementaciones de referencia (abrir solo **después** de intentarlo)

| Recurso | Útil en fases |
|---|---|
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | 00 |
| [karpathy/makemore](https://github.com/karpathy/makemore) | 01–03 |
| [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | 04–05 |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 05, 07 |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) (incluye `exercise.md`, úsalo como guía) | 06 |
| [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt): un commit por paso, ideal como solucionario incremental | 07 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c): GPT-2 en C/CUDA puro | 07–08 |
| [karpathy/nanochat](https://github.com/karpathy/nanochat): pipeline completo y moderno de un chat LLM pequeño | 09+ |
| [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt): *speedrun* de GPT-2 con trucos modernos | 09 |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (libro de S. Raschka) | 04–09 |
| [Stanford CS336, Assignment 1](https://github.com/stanford-cs336/assignment1-basics) ([handout PDF](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)): tokenizador, Transformer y AdamW **sin usar `torch.nn`**. Es un "modo difícil" excelente | 06–09 |
| [Train a miniGPT with JAX](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html) (JAX AI Stack) | 11 |

Curso completo recomendado como columna vertebral: Karpathy,
[**Neural Networks: Zero to Hero**](https://karpathy.ai/zero-to-hero.html) (las fases 00–07
siguen su orden). Para ir más hondo: [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/).

---

## 7. Datos

Ver [data/README.md](data/README.md). En resumen: `names.txt` (fases 01–03),
Tiny Shakespeare (04–05, 10–11), TinyStories (07–09) y HellaSwag (evaluación, fase 07).
