# Tu cara haciendo ejercicio, con los músculos en rojo

Genera una foto fotorrealista de **la persona de tu foto haciendo sentadilla con barra**
y marca en rojo los músculos que trabajan:

| | Músculo | Papel |
|---|---|---|
| 1 | Cuádriceps | motor principal (rojo intenso) |
| 2 | Glúteo mayor | motor principal (rojo intenso) |
| 3 | Aductores | motor principal (rojo intenso) |
| 4 | Isquiotibiales | sinergista (rojo suave) |
| 5 | Erectores espinales | estabilizador (rojo suave) |
| 6 | Abdominales / core | estabilizador (rojo suave) |
| 7 | Gemelos y sóleo | estabilizador (rojo suave) |

Cómo funciona:

1. **InstantID** (en Replicate) conserva tu cara a partir de una sola foto.
2. **MediaPipe Pose** detecta el cuerpo en la imagen generada (33 puntos + silueta).
3. Cada músculo se dibuja como una zona anatómica entre articulaciones, recortada con la
   silueta para que el rojo no se salga del cuerpo. Se adapta a la vista: lateral
   (cuádriceps, glúteo, isquios, erectores, core, gemelos), de frente (cuádriceps,
   aductores, core) o de espaldas.
4. Se añade un panel lateral con la leyenda numerada.

## Instalación

```bash
cd ejercicios
pip install -r requirements.txt
export REPLICATE_API_TOKEN="tu_token"        # https://replicate.com/account/api-tokens
# Windows PowerShell:  $env:REPLICATE_API_TOKEN="tu_token"
```

Los modelos de MediaPipe (~30 MB) se descargan solos la primera vez en `ejercicios/modelos/`.

## Uso

```bash
# Tú haciendo sentadilla con barra + músculos en rojo
python generar_ejercicios.py --foto mi_cara.jpg

# Describe a la persona (en inglés): InstantID copia la cara, pero no el pelo ni el cuerpo
python generar_ejercicios.py --foto mi_cara.jpg --descripcion "a young woman with long straight dark brown hair"

# Mejor resultado: foto de referencia con la postura (de cualquier persona, cara visible)
python generar_ejercicios.py --foto mi_cara.jpg --pose referencia_sentadilla.jpg

# Solo marcar los músculos en una foto que ya tienes (no usa Replicate ni token)
python generar_ejercicios.py --marcar foto_haciendo_sentadilla.jpg

# Opciones útiles
python generar_ejercicios.py --foto mi_cara.jpg --variantes 4 --semilla 42
python generar_ejercicios.py --foto mi_cara.jpg --sin-leyenda      # solo la foto marcada
python generar_ejercicios.py --foto mi_cara.jpg --sin-musculos     # solo la foto
```

Salida en `resultados/`:

- `sentadilla_1.png` – la foto generada, limpia
- `sentadilla_1_musculos.png` – la misma con los músculos marcados y la leyenda

## Consejos

- **Foto de la cara**: de frente, nítida, bien iluminada, sin gafas de sol.
- **Cara y cuerpo**: InstantID solo copia la cara; el cuerpo, el pelo y la ropa salen del
  texto (`--descripcion`). Si quieres que también el cuerpo sea el real, usa una foto real
  de la persona haciendo la sentadilla con `--marcar`.
- **Cuerpo entero**: InstantID pone la cara del resultado donde está en la foto de entrada,
  así que un selfie daría un primer plano. Sin `--pose` el script reduce tu cara y la coloca
  arriba en un lienzo vertical para que salga el cuerpo entero. Con `--pose` la postura y el
  encuadre salen de la foto de referencia, que es lo más fiable.
- **Marcado**: funciona mejor con el cuerpo entero visible y vista lateral o de tres cuartos.
  Los músculos que no se ven desde ese ángulo aparecen en la leyenda como «no visible».
- Los demás ejercicios (`press_banca`, `dominadas`, …) siguen generándose, pero todavía no
  tienen mapa muscular: se guarda solo la foto. Para añadir uno, crea su entrada en `MAPAS`
  dentro de `musculos.py` reutilizando las zonas existentes.
