# Estudio de canciones

Crea canciones completas con voz y letra, en cualquier idioma, en tu propio PC. Gratis, sin cuentas y sin límites.

Es una versión optimizada de [HeartMuLa](https://github.com/HeartMuLa/heartlib), el modelo de música abierto, inspirada en [HeartMuLa-Studio](https://github.com/fspecii/HeartMuLa-Studio). El motor de inferencia está reescrito para que gaste mucho menos y vaya bastante más rápido, sin perder calidad (está medido; lo tienes abajo).

## Instalación (Windows)

1. Descarga el ZIP (botón verde **Code → Download ZIP**) y descomprímelo, por ejemplo en `C:\EstudioCanciones`.
2. Doble clic en **`Instalar.bat`**. Instala Python, PyTorch y los modelos (~7 GB) dentro de la carpeta y no toca nada del sistema. Si se corta, vuelve a ejecutarlo y continúa.
3. Opcional y recomendado: **`Instalar pack calidad (mejor de 4).bat`** (+2 GB).
4. Doble clic en **`Abrir Estudio.bat`**: se abre la app en el navegador.

Las canciones se guardan en `canciones` (MP3 y un `.txt` con la letra, el estilo y la semilla). Hay más trucos en `LEEME.txt`.

### Qué PC hace falta

| Equipo | Funciona | Notas |
|---|---|---|
| NVIDIA RTX 30/40/50 con ≥ 8 GB | Sí | Hasta 4 tomas a la vez |
| NVIDIA con 6-8 GB | Sí | Hasta 2 tomas; pico medido de 4,2 GB (1 toma) y 4,4 GB (2) |
| RTX 20xx | Debería | El modo fp16 está validado numéricamente, pero no lo he podido probar en una RTX 20 real |
| GTX 16xx o anteriores | Sin probar | |
| Sin NVIDIA (AMD, Intel, solo CPU) | Sí, lento | ~5 min por minuto de canción en un Core Ultra 5 245K; necesita 16 GB de RAM |

Disco: unos 12 GB (14 GB con el pack de calidad).

## Qué se ha optimizado

Medido en una RTX 5060 Ti de 16 GB con la misma canción de 60 s:

| | HeartMuLa-Studio (original) | Este estudio |
|---|---|---|
| Descarga de modelos | 22,3 GB (fp32) | 7,1 GB |
| VRAM | ~14 GB de pico | ~4 GB generando; 4,2 GB de pico en gráficas pequeñas |
| RAM | ~22 GB de pico | ~4,5 GB |
| 1 toma de 60 s | 62 s | 17-20 s |
| 4 tomas de 60 s, escuchadas y ordenadas | no existe | ~51 s |
| Canción de 4 min, 4 tomas | – | 2 min 39 s |
| Instalación | Docker de ~10 GB, o Python + Node | un `.bat` |

Cómo:

- **Transformer en int8 con un kernel Triton propio.** Pesos por canal en 8 bits, unas 2 veces más rápido que fp16 en tamaños pequeños de lote. También van en int8 los embeddings y las cabezas.
- **CUDA graphs.** Cada fotograma (backbone de 3B y 7 pasadas del decoder) se graba como un único grafo, con caché KV estática por tramos y GQA sin copiar la caché.
- **fp16 en vez de bf16.** Medido: queda más cerca del original (KL 0,0015 frente a 0,0023) y funciona en GPUs sin bf16.
- **Códec sin el encoder**, que no se usa para generar, y con el **decodificador de onda por trozos** en gráficas pequeñas: el pico de esa fase baja de 2,4 GB a 0,3 GB con el mismo resultado.
- **Embeddings de texto en la CPU**, porque solo se usan en el prompt.
- **Turnos de VRAM automáticos** en gráficas de menos de 12 GB: transformer, códec y jueces se turnan en la GPU.
- **Varias tomas en el mismo lote.** Generar 4 tomas cuesta un 10 % más que generar una.

## Calidad

No se ha dado nada por bueno sin medirlo:

- **Divergencia del modelo**: con *teacher forcing* sobre 60 s de canción en 3 estilos, la KL frente al original es de ~0,003 y no se acumula con el tiempo. Es del orden del propio ruido numérico (0,0013). El int4 se descartó porque daba 0,03, diez veces más.
- **Jueces automáticos**: [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) de Meta para el sonido, y [HeartTranscriptor](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss) más Demucs para comprobar si la letra se entiende. Resultados:
  - 25 canciones del original frente a 25 de este estudio: calidad de producción 8,26 en ambos.
  - 100 frente a 100 en int8 y en el equivalente exacto del original: error de letra del 3,5 % frente al 3,7 %, sin diferencia.
- **Códec**: comparando precisiones en 20 canciones, las puntuaciones salen iguales (±0,01). Subir los pasos de 10 a 20 o 32 no mejora nada.
- **Parámetros**: los de los autores (CFG 1,5 y temperatura 1,0) son los mejores de los que probé. Subir el CFG empeora la letra.

**Mejor de 4** (con el pack de calidad): HeartMuLa a veces suelta una toma en la que la letra se pierde (~5 % de las veces, también en el original). Con 4 tomas, la app las escucha todas y te pone primero la que mejor suena y mejor se entiende. En las pruebas dejó la letra al 100 % en todos los estilos.

## Licencias

- Código de este repositorio: MIT.
- `app/heartcodec/`: código de [heartlib](https://github.com/HeartMuLa/heartlib), Apache 2.0.
- Modelos HeartMuLa, HeartCodec y HeartTranscriptor (HeartMuLa Team): Apache 2.0. Los pesos del Release son conversiones de los oficiales: int8 y fp16 en el caso del transcriptor. Si el Release no está disponible, el instalador baja los oficiales de Hugging Face y los convierte en tu PC.
- Audiobox Aesthetics (Meta): CC BY 4.0. Solo se descarga con el pack de calidad.
