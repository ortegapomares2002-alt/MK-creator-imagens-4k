"""
Genera imágenes fotorrealistas de la persona de TU foto haciendo distintos ejercicios
y marca en rojo los músculos que trabajan.

1. La imagen se crea con uno de dos motores:
   - gemini (por defecto): Gemini ve la foto entera y conserva la cara, el pelo y la
     complexión de la persona (lo más parecido a su cuerpo real).
   - instantid (Replicate): conserva solo la cara; el cuerpo sale del texto.
2. MediaPipe detecta la pose en la imagen generada y pinta los músculos implicados
   (rojo intenso = motores principales, rojo suave = sinergistas / estabilizadores).

Requisitos:
    pip install -r requirements.txt
    export GEMINI_API_KEY="tu_clave"          # https://aistudio.google.com/apikey
    export REPLICATE_API_TOKEN="tu_token"     # solo para --motor instantid
    (Windows PowerShell:  $env:GEMINI_API_KEY="tu_clave")

Uso:
    # La persona de la foto (cara y cuerpo) haciendo sentadilla con barra, músculos en rojo
    python generar_ejercicios.py --foto foto.jpg

    # Copiar además la postura y el ángulo de otra foto (de cualquier persona)
    python generar_ejercicios.py --foto foto.jpg --pose referencia_sentadilla.jpg

    # Con InstantID en Replicate (solo conserva la cara)
    python generar_ejercicios.py --foto mi_cara.jpg --motor instantid

    # Solo marcar los músculos en una foto que ya tienes (no usa Replicate)
    python generar_ejercicios.py --marcar foto_haciendo_sentadilla.jpg

    # Otros ejercicios (sin mapa muscular todavía: se guarda solo la foto)
    python generar_ejercicios.py --foto mi_cara.jpg --ejercicios dominadas flexiones
"""

import argparse
import io
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageOps

from musculos import MAPAS, detectar_cara, marcar_musculos

MODELO = "zsxkib/instant-id"
# Modelos de imagen de Gemini, de más a menos fiel; si uno no está disponible se usa el siguiente
MODELOS_GEMINI = ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "gemini-2.5-flash-image"]

# Descripción de cada ejercicio (en inglés: el modelo responde mejor)
EJERCICIOS = {
    "sentadilla":     "doing a barbell back squat, full body shot from head to feet, "
                      "three-quarter side view, deep squat position with thighs parallel to the floor, "
                      "loaded olympic barbell resting on the upper back, hands gripping the bar, "
                      "wearing a fitted sports top and athletic shorts, bare legs",
    "press_banca":    "lying on a bench doing a barbell bench press, arms extended",
    "peso_muerto":    "performing a deadlift, lifting a loaded barbell from the floor, straight back",
    "dominadas":      "doing pull-ups on a pull-up bar, chin above the bar, back muscles engaged",
    "flexiones":      "doing push-ups on the gym floor, body straight, side view",
    "curl_biceps":    "doing a dumbbell biceps curl, standing, flexed biceps",
    "press_militar":  "doing an overhead shoulder press with dumbbells, standing",
    "zancadas":       "doing walking lunges with dumbbells in each hand",
    "remo":           "doing a bent-over barbell row, back flat",
    "plancha":        "holding a forearm plank on a yoga mat, core engaged",
    "abdominales":    "doing crunches on a mat, abdominal muscles contracted",
    "fondos":         "doing triceps dips on parallel bars",
}

PROMPT_BASE = (
    "photorealistic photo of {persona}, {ejercicio}, in a modern gym, "
    "athletic sportswear, natural gym lighting, sweat, realistic skin texture, "
    "natural body proportions, "
    "sharp focus, shot on a full-frame DSLR, 50mm lens, high detail, 8k"
)

NEGATIVE = (
    "cartoon, anime, illustration, painting, 3d render, cgi, plastic skin, "
    "deformed hands, extra fingers, extra limbs, distorted face, blurry, "
    "low quality, watermark, text, bad anatomy, close-up, headshot, cropped body, out of frame"
)

PROMPT_GEMINI = (
    "Photorealistic photo of the same person shown in the reference photo{extra}, {ejercicio}, "
    "in a modern gym, natural gym lighting, realistic skin texture, sharp focus, shot on a full-frame DSLR. "
    "Keep the person's identity exactly: same face, facial features, hair, skin tone, and the same body "
    "build, weight and proportions as in the reference photo. Do not make the body slimmer, more muscular "
    "or idealized. Do not keep the outfit, hat or objects from the reference photo. "
    "No text, no watermark.{pose}"
)
PROMPT_GEMINI_POSE = (
    " A second image is attached: copy only the body pose and camera angle from it. It shows a "
    "different person, so never use it for the face or the body."
)

# Lienzo vertical para el encuadre de cuerpo entero (múltiplos de 64, ~1 Mpx como SDXL)
LIENZO = (896, 1152)
ANCHO_CARA = 0.12          # ancho de la cara respecto al lienzo
CENTRO_CARA = (0.50, 0.16)  # posición del centro de la cara en el lienzo


def version_actual() -> str:
    """Obtiene la última versión publicada del modelo."""
    import replicate

    modelo = replicate.models.get(MODELO)
    return f"{MODELO}:{modelo.latest_version.id}"


def descargar(url: str, destino: Path) -> None:
    import requests

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    destino.write_bytes(r.content)


def encuadre_cuerpo_entero(foto: Path) -> Image.Image:
    """
    InstantID coloca la cara del resultado en la misma posición y tamaño que en la
    foto de entrada: con un selfie sale un primer plano. Para que salga el cuerpo
    entero se reduce la cara y se coloca arriba en un lienzo vertical.
    """
    img = ImageOps.exif_transpose(Image.open(foto)).convert("RGB")
    W, H = LIENZO
    try:
        caja = detectar_cara(img)
    except Exception as e:  # p. ej. mediapipe no instalado
        print(f"   aviso: no se pudo detectar la cara ({e})")
        caja = None
    if caja is None:
        print("   aviso: no se encuentra la cara en la foto; se supone centrada")
        x, y, w, h = img.width * 0.3, img.height * 0.2, img.width * 0.4, img.height * 0.4
    else:
        x, y, w, h = caja
    escala = ANCHO_CARA * W / w

    # Recorte generoso alrededor de la cara (pelo, orejas, cuello), sin salirse de la foto
    m = 0.9
    izq, arr = max(0, int(x - w * m)), max(0, int(y - h * m))
    der, aba = min(img.width, int(x + w * (1 + m))), min(img.height, int(y + h * (1 + m)))
    recorte = img.crop((izq, arr, der, aba))
    recorte = recorte.resize((max(1, round(recorte.width * escala)), max(1, round(recorte.height * escala))),
                             Image.LANCZOS)

    # Se pega de modo que el centro de la cara caiga en CENTRO_CARA
    lienzo = Image.new("RGB", LIENZO, (128, 128, 128))
    cx, cy = CENTRO_CARA[0] * W, CENTRO_CARA[1] * H
    lienzo.paste(recorte, (round(cx - (x + w / 2 - izq) * escala), round(cy - (y + h / 2 - arr) * escala)))
    return lienzo


def _config_gemini(types, modelo):
    opciones = {"aspect_ratio": "3:4"}
    if modelo.startswith("gemini-3"):
        opciones["image_size"] = "2K"
    try:
        return types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"],
                                           image_config=types.ImageConfig(**opciones))
    except Exception:  # SDK antiguo sin ImageConfig
        return types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])


def _imagen_gemini(cliente, types, modelos, contenido) -> bytes:
    """Pide una imagen probando los modelos en orden; quita de la lista los que fallan."""
    errores = []
    while modelos:
        modelo = modelos[0]
        try:
            resp = cliente.models.generate_content(model=modelo, contents=contenido,
                                                   config=_config_gemini(types, modelo))
        except Exception as e:
            errores.append(f"{modelo}: {e}")
            modelos.pop(0)
            continue
        candidato = resp.candidates[0] if resp.candidates else None
        partes = candidato.content.parts if candidato and candidato.content and candidato.content.parts else []
        for parte in partes:
            if parte.inline_data and parte.inline_data.data:
                return parte.inline_data.data
        texto = " ".join(p.text for p in partes if getattr(p, "text", None))
        motivo = (getattr(candidato, "finish_reason", None) if candidato
                  else getattr(resp.prompt_feedback, "block_reason", None))
        raise RuntimeError(f"{modelo} no devolvió imagen ({motivo}). {texto}".strip())
    raise RuntimeError("ningún modelo de Gemini respondió:\n     " + "\n     ".join(errores))


def generar_gemini(foto: Path, ejercicio: str, variantes: int, salida: Path, modelos: list,
                   pose: Path = None, marcar: bool = True, leyenda: bool = True, descripcion: str = None):
    from google import genai
    from google.genai import types

    cliente = genai.Client()  # lee GEMINI_API_KEY (o GOOGLE_API_KEY)
    prompt = PROMPT_GEMINI.format(extra=f" ({descripcion})" if descripcion else "",
                                  ejercicio=EJERCICIOS[ejercicio],
                                  pose=PROMPT_GEMINI_POSE if pose else "")
    contenido = [prompt, ImageOps.exif_transpose(Image.open(foto)).convert("RGB")]
    if pose:
        contenido.append(ImageOps.exif_transpose(Image.open(pose)).convert("RGB"))

    print(f"→ Generando '{ejercicio}' con Gemini...")
    for i in range(1, variantes + 1):
        datos = _imagen_gemini(cliente, types, modelos, contenido)
        destino = salida / f"{ejercicio}_{i}.png"
        Image.open(io.BytesIO(datos)).save(destino)
        print(f"   guardada: {destino}  ({modelos[0]})")
        if marcar and ejercicio in MAPAS:
            marcar_imagen(destino, ejercicio, leyenda)


def generar(foto: Path, ejercicio: str, variantes: int, semilla, salida: Path, ref: str,
            pose: Path = None, marcar: bool = True, leyenda: bool = True,
            descripcion: str = None):
    import replicate

    prompt = PROMPT_BASE.format(persona=descripcion or "the person", ejercicio=EJERCICIOS[ejercicio])
    with ExitStack() as pila:
        entrada = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE,
            "num_outputs": variantes,
            "num_inference_steps": 30,
            "guidance_scale": 5,
            # Fuerza de la identidad facial (más alto = más parecido, menos flexible)
            "ip_adapter_scale": 0.8,
            "controlnet_conditioning_scale": 0.8,
            "output_format": "png",
        }
        if pose:
            # La postura (y la posición de la cara) salen de la foto de referencia
            entrada["image"] = pila.enter_context(open(foto, "rb"))
            entrada["pose_image"] = pila.enter_context(open(pose, "rb"))
            entrada["enable_pose_controlnet"] = True
            entrada["pose_strength"] = 0.8
        else:
            tmp = Path(pila.enter_context(tempfile.TemporaryDirectory())) / "encuadre.png"
            encuadre_cuerpo_entero(foto).save(tmp)
            entrada["image"] = pila.enter_context(open(tmp, "rb"))
            # Sin referencia no hay esqueleto útil: que el cuerpo lo decida el prompt
            entrada["enable_pose_controlnet"] = False
        if semilla is not None:
            entrada["seed"] = semilla

        print(f"→ Generando '{ejercicio}'...")
        resultado = replicate.run(ref, input=entrada)
    if not isinstance(resultado, list):
        resultado = [resultado]

    for i, item in enumerate(resultado, 1):
        destino = salida / f"{ejercicio}_{i}.png"
        if hasattr(item, "read"):          # FileOutput (cliente nuevo)
            destino.write_bytes(item.read())
        else:                               # URL (cliente antiguo)
            descargar(str(item), destino)
        print(f"   guardada: {destino}")
        if marcar and ejercicio in MAPAS:
            marcar_imagen(destino, ejercicio, leyenda)


def marcar_imagen(origen: Path, ejercicio: str, leyenda: bool = True, salida: Path = None) -> None:
    destino = (salida or origen.parent) / f"{origen.stem}_musculos.png"
    try:
        img = ImageOps.exif_transpose(Image.open(origen))
        marcar_musculos(img, ejercicio, leyenda=leyenda).save(destino)
        print(f"   músculos marcados: {destino}")
    except Exception as e:
        print(f"   no se pudieron marcar los músculos en {origen.name}: {e}")


def main():
    p = argparse.ArgumentParser(description="Tu cara haciendo ejercicios, fotorrealista, con los músculos en rojo.")
    origen = p.add_mutually_exclusive_group(required=True)
    origen.add_argument("--foto", help="Foto de la persona, con la cara de frente y nítida")
    origen.add_argument("--marcar", nargs="+", metavar="IMAGEN",
                        help="No genera nada: solo marca los músculos en estas fotos de cuerpo entero")
    p.add_argument("--ejercicios", nargs="*", default=["sentadilla"],
                   help=f"Opciones: {', '.join(EJERCICIOS)} (por defecto: sentadilla)")
    p.add_argument("--motor", choices=["gemini", "instantid"], default="gemini",
                   help="gemini: conserva cara y complexión (por defecto). instantid: solo la cara")
    p.add_argument("--modelo", help=f"Modelo de Gemini concreto (por defecto prueba: {', '.join(MODELOS_GEMINI)})")
    p.add_argument("--descripcion",
                   help="Cómo es la persona, en inglés (sobre todo con instantid, que no copia pelo ni cuerpo). "
                        "Ej.: \"a young woman with long straight dark brown hair\"")
    p.add_argument("--pose", help="Foto de referencia de cuerpo entero con la postura deseada "
                                  "(de cualquier persona, con la cara visible). Muy recomendable")
    p.add_argument("--variantes", type=int, default=2, help="Imágenes por ejercicio (1-4)")
    p.add_argument("--semilla", type=int, default=None, help="Para resultados repetibles (solo instantid)")
    p.add_argument("--salida", default="resultados", help="Carpeta de salida")
    p.add_argument("--sin-musculos", action="store_true", help="No marcar los músculos")
    p.add_argument("--sin-leyenda", action="store_true", help="Marcar los músculos sin el panel lateral")
    args = p.parse_args()

    desconocidos = [e for e in args.ejercicios if e not in EJERCICIOS]
    if desconocidos:
        sys.exit(f"Ejercicios no válidos: {desconocidos}. Opciones: {list(EJERCICIOS)}")

    salida = Path(args.salida)
    salida.mkdir(exist_ok=True)

    if args.marcar:
        if len(args.ejercicios) != 1 or args.ejercicios[0] not in MAPAS:
            sys.exit(f"Con --marcar indica un solo ejercicio con mapa muscular: {list(MAPAS)}")
        for ruta in map(Path, args.marcar):
            if not ruta.exists():
                print(f"   no encuentro: {ruta}")
                continue
            marcar_imagen(ruta, args.ejercicios[0], not args.sin_leyenda, salida)
        print(f"\nListo. Imágenes en: {salida.resolve()}")
        return

    if args.motor == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Falta GEMINI_API_KEY. Consíguela en https://aistudio.google.com/apikey")
    if args.motor == "instantid" and not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("Falta REPLICATE_API_TOKEN. Consíguelo en replicate.com/account/api-tokens")

    foto = Path(args.foto)
    if not foto.exists():
        sys.exit(f"No encuentro la foto: {foto}")
    pose = Path(args.pose) if args.pose else None
    if pose and not pose.exists():
        sys.exit(f"No encuentro la foto de referencia: {pose}")
    if pose and len(args.ejercicios) > 1:
        sys.exit("--pose es la postura de un ejercicio concreto: úsalo con un solo ejercicio")

    sin_mapa = [e for e in args.ejercicios if e not in MAPAS]
    if sin_mapa and not args.sin_musculos:
        print(f"Aviso: sin mapa muscular para {sin_mapa}; de esos se guarda solo la foto.")

    variantes = max(1, min(args.variantes, 4))
    opciones = dict(pose=pose, marcar=not args.sin_musculos, leyenda=not args.sin_leyenda,
                    descripcion=args.descripcion)
    if args.motor == "gemini":
        modelos = [args.modelo] if args.modelo else list(MODELOS_GEMINI)
    else:
        ref = version_actual()
    for ej in args.ejercicios:
        try:
            if args.motor == "gemini":
                generar_gemini(foto, ej, variantes, salida, modelos, **opciones)
            else:
                generar(foto, ej, variantes, args.semilla, salida, ref, **opciones)
        except Exception as e:
            print(f"   error en '{ej}': {e}")

    print(f"\nListo. Imágenes en: {salida.resolve()}")


if __name__ == "__main__":
    main()
