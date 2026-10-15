"""
Marca en rojo los músculos que trabajan en un ejercicio sobre una foto.

Detecta la pose del cuerpo con MediaPipe (33 puntos + silueta), dibuja cada
músculo como una zona anatómica entre articulaciones y la recorta con la
silueta de la persona, de modo que el rojo nunca se sale del cuerpo.

  rojo intenso -> motores principales
  rojo suave   -> sinergistas / estabilizadores

Uso independiente (sin Replicate):
    python musculos.py foto_sentadilla.jpg
    python musculos.py foto_sentadilla.jpg --salida marcada.png --sin-leyenda
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
URL_MODELO_POSE = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
URL_MODELO_CARA = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

# Índices de MediaPipe Pose
NARIZ, OREJA_I, OREJA_D = 0, 7, 8
HOMBRO = {"i": 11, "d": 12}
CADERA = {"i": 23, "d": 24}
RODILLA = {"i": 25, "d": 26}
TOBILLO = {"i": 27, "d": 28}
TALON = {"i": 29, "d": 30}
PUNTA = {"i": 31, "d": 32}

ROJO = (225, 20, 30)
ALFA = {"principal": 0.62, "secundario": 0.34}
COLOR_ROL = {"principal": ROJO, "secundario": (235, 110, 110)}
ROL_TEXTO = {"principal": "Motor principal", "secundario": "Sinergista / estabilizador"}


@dataclass
class Musculo:
    nombre: str
    rol: str    # "principal" | "secundario"
    zona: str   # clave de ZONAS


MAPAS = {
    "sentadilla": {
        "titulo": "Sentadilla con barra",
        "musculos": [
            Musculo("Cuádriceps", "principal", "cuadriceps"),
            Musculo("Glúteo mayor", "principal", "gluteos"),
            Musculo("Aductores", "principal", "aductores"),
            Musculo("Isquiotibiales", "secundario", "isquiotibiales"),
            Musculo("Erectores espinales", "secundario", "erectores"),
            Musculo("Abdominales / core", "secundario", "abdominales"),
            Musculo("Gemelos y sóleo", "secundario", "gemelos"),
        ],
    },
}


# --------------------------------------------------------------------------- #
#  Modelos y detección
# --------------------------------------------------------------------------- #

def _modelo(url: str) -> str:
    destino = CARPETA_MODELOS / url.rsplit("/", 1)[-1]
    if not destino.exists():
        CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
        print(f"   descargando modelo {destino.name} (solo la primera vez)...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        tmp = destino.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(destino)
    return str(destino)


def _mp_image(img: Image.Image):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img.convert("RGB"))))


def detectar_cara(img: Image.Image):
    """Devuelve la caja (x, y, ancho, alto) en píxeles de la cara más grande, o None."""
    from mediapipe.tasks.python import BaseOptions, vision

    opciones = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_modelo(URL_MODELO_CARA)),
        min_detection_confidence=0.5,
    )
    with vision.FaceDetector.create_from_options(opciones) as detector:
        caras = detector.detect(_mp_image(img)).detections
    if not caras:
        return None
    b = max(caras, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
    return b.origin_x, b.origin_y, b.width, b.height


class Pose:
    """Puntos de la pose en píxeles + silueta + vista (lateral / frontal / espalda)."""

    def __init__(self, img: Image.Image):
        from mediapipe.tasks.python import BaseOptions, vision

        opciones = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_modelo(URL_MODELO_POSE)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            output_segmentation_masks=True,
        )
        with vision.PoseLandmarker.create_from_options(opciones) as detector:
            res = detector.detect(_mp_image(img))
        if not res.pose_landmarks:
            raise ValueError("no se detecta ninguna persona de cuerpo entero en la imagen")

        self.w, self.h = img.size
        lm = res.pose_landmarks[0]
        self.p = np.array([[l.x * self.w, l.y * self.h] for l in lm])
        self.z = np.array([l.z for l in lm])
        self.vis = np.array([l.visibility or 0.0 for l in lm])

        mascara = res.segmentation_masks[0].numpy_view()
        self.silueta = np.clip(mascara.reshape(self.h, self.w).astype(np.float32), 0, 1)

        self._clasificar_vista()

    def _clasificar_vista(self):
        hi, hd = self.p[HOMBRO["i"]], self.p[HOMBRO["d"]]
        ci, cd = self.p[CADERA["i"]], self.p[CADERA["d"]]
        self.tronco = float(np.linalg.norm((hi + hd) / 2 - (ci + cd) / 2)) or 1.0
        apertura = abs(hi[0] - hd[0]) / self.tronco

        if apertura < 0.45:
            self.vista = "lateral"
            # Hacia dónde mira: dirección talón -> punta del pie (respaldo: nariz vs orejas)
            pies = sum(
                (self.p[PUNTA[l]][0] - self.p[TALON[l]][0]) * max(self.vis[PUNTA[l]], 0.1)
                for l in "id"
            )
            if abs(pies) < 1e-3:
                pies = self.p[NARIZ][0] - (self.p[OREJA_I][0] + self.p[OREJA_D][0]) / 2
            self.mira = 1 if pies >= 0 else -1
            # Pierna más cercana a la cámara (z menor = más cerca)
            zi = self.z[[CADERA["i"], RODILLA["i"], TOBILLO["i"]]].mean()
            zd = self.z[[CADERA["d"], RODILLA["d"], TOBILLO["d"]]].mean()
            self.lados = ["i" if zi <= zd else "d"]
        else:
            # De frente el hombro izquierdo queda a la derecha de la imagen; de espaldas, al revés
            self.vista = "frontal" if hi[0] > hd[0] else "espalda"
            self.mira = 0
            self.lados = ["i", "d"]

    def anterior(self, a, b):
        """Normal unitaria hacia la cara anterior del segmento a->b (proximal -> distal), vista lateral."""
        d = (b - a) / (np.linalg.norm(b - a) or 1.0)
        n = np.array([d[1], -d[0]])
        return n * self.mira


# --------------------------------------------------------------------------- #
#  Geometría de las zonas musculares
# --------------------------------------------------------------------------- #

def _unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def _banda(a, b, t0, t1, n, centro, semiancho, afilado=0.7, pasos=28):
    """Forma de vientre muscular a lo largo de a->b, desplazada `centro` px según la normal n."""
    arriba, abajo = [], []
    for i in range(pasos + 1):
        s = i / pasos
        t = t0 + (t1 - t0) * s
        perfil = math.sin(math.pi * (0.06 + 0.88 * s)) ** afilado
        c = a + (b - a) * t + n * centro
        arriba.append(c + n * semiancho * perfil)
        abajo.append(c - n * semiancho * perfil)
    return arriba + abajo[::-1]


def _elipse(c, rx, ry, eje, pasos=40):
    eje = _unit(eje)
    perp = np.array([-eje[1], eje[0]])
    return [c + eje * rx * math.cos(2 * math.pi * k / pasos) + perp * ry * math.sin(2 * math.pi * k / pasos)
            for k in range(pasos)]


def _pierna(pose, lado):
    return pose.p[CADERA[lado]], pose.p[RODILLA[lado]], pose.p[TOBILLO[lado]]


def _medial(pose, lado):
    """Normal del muslo que apunta hacia la otra pierna (vista frontal / espalda)."""
    cad, rod, _ = _pierna(pose, lado)
    d = _unit(rod - cad)
    n = np.array([-d[1], d[0]])
    otra = pose.p[CADERA["d" if lado == "i" else "i"]]
    return n if np.dot(otra - cad, n) > 0 else -n


def _ancho_muslo(pose):
    if pose.vista == "lateral":
        cad, rod, _ = _pierna(pose, pose.lados[0])
        return 0.42 * np.linalg.norm(rod - cad)
    return 0.62 * np.linalg.norm(pose.p[CADERA["i"]] - pose.p[CADERA["d"]])


def zona_cuadriceps(pose):
    if pose.vista == "espalda":
        return []
    W = _ancho_muslo(pose)
    polis = []
    for lado in pose.lados:
        cad, rod, _ = _pierna(pose, lado)
        if pose.vista == "lateral":
            n = pose.anterior(cad, rod)
            polis.append(_banda(cad, rod, 0.10, 0.93, n, 0.38 * W, 0.42 * W, 0.6))
        else:
            d = _unit(rod - cad)
            n = np.array([-d[1], d[0]])
            polis.append(_banda(cad, rod, 0.10, 0.95, n, 0, 0.55 * W, 0.5))
    return polis


def zona_isquiotibiales(pose):
    if pose.vista == "frontal":
        return []
    W = _ancho_muslo(pose)
    polis = []
    for lado in pose.lados:
        cad, rod, _ = _pierna(pose, lado)
        if pose.vista == "lateral":
            n = -pose.anterior(cad, rod)
            polis.append(_banda(cad, rod, 0.22, 0.90, n, 0.36 * W, 0.40 * W, 0.8))
        else:
            d = _unit(rod - cad)
            n = np.array([-d[1], d[0]])
            polis.append(_banda(cad, rod, 0.20, 0.92, n, 0, 0.50 * W, 0.7))
    return polis


def zona_gluteos(pose):
    if pose.vista == "frontal":
        return []
    W = _ancho_muslo(pose)
    if pose.vista == "lateral":
        lado = pose.lados[0]
        cad, rod, _ = _pierna(pose, lado)
        hom = pose.p[HOMBRO[lado]]
        atras = _unit(-pose.anterior(cad, rod) - pose.anterior(hom, cad))
        return [_elipse(cad + atras * 0.42 * W, 0.70 * W, 0.62 * W, atras)]
    polis = []
    for lado in pose.lados:
        cad, rod, _ = _pierna(pose, lado)
        hacia_arriba = _unit(cad - rod)
        centro = cad + hacia_arriba * 0.05 * W + _medial(pose, lado) * 0.15 * W
        polis.append(_elipse(centro, 0.62 * W, 0.55 * W, hacia_arriba))
    return polis


def zona_aductores(pose):
    # Están en la cara interna del muslo: en vista lateral los tapa la propia pierna
    if pose.vista == "lateral":
        return []
    W = _ancho_muslo(pose)
    polis = []
    for lado in pose.lados:
        cad, rod, _ = _pierna(pose, lado)
        polis.append(_banda(cad, rod, 0.04, 0.62, _medial(pose, lado), 0.42 * W, 0.30 * W, 0.8))
    return polis


def zona_gemelos(pose):
    if pose.vista == "frontal":
        return []
    polis = []
    for lado in pose.lados:
        _, rod, tob = _pierna(pose, lado)
        L = np.linalg.norm(tob - rod)
        if pose.vista == "lateral":
            n = -pose.anterior(rod, tob)
            polis.append(_banda(rod, tob, 0.05, 0.72, n, 0.20 * L, 0.20 * L, 0.9))
        else:
            d = _unit(tob - rod)
            n = np.array([-d[1], d[0]])
            polis.append(_banda(rod, tob, 0.06, 0.70, n, 0, 0.16 * L, 0.8))
    return polis


def _tronco(pose, lado=None):
    if lado:
        return pose.p[HOMBRO[lado]], pose.p[CADERA[lado]]
    hom = (pose.p[HOMBRO["i"]] + pose.p[HOMBRO["d"]]) / 2
    cad = (pose.p[CADERA["i"]] + pose.p[CADERA["d"]]) / 2
    return hom, cad


def zona_erectores(pose):
    if pose.vista == "frontal":
        return []
    T = pose.tronco
    if pose.vista == "lateral":
        hom, cad = _tronco(pose, pose.lados[0])
        n = -pose.anterior(hom, cad)
        return [_banda(hom, cad, 0.30, 1.00, n, 0.20 * T, 0.10 * T, 0.5)]
    hom, cad = _tronco(pose)
    d = _unit(cad - hom)
    n = np.array([-d[1], d[0]])
    return [_banda(hom, cad, 0.25, 1.00, n, s * 0.09 * T, 0.06 * T, 0.5) for s in (-1, 1)]


def zona_abdominales(pose):
    if pose.vista == "espalda":
        return []
    T = pose.tronco
    if pose.vista == "lateral":
        hom, cad = _tronco(pose, pose.lados[0])
        n = pose.anterior(hom, cad)
        return [_banda(hom, cad, 0.40, 0.92, n, 0.20 * T, 0.10 * T, 0.5)]
    hom, cad = _tronco(pose)
    d = _unit(cad - hom)
    n = np.array([-d[1], d[0]])
    return [_banda(hom, cad, 0.38, 0.95, n, 0, 0.17 * T, 0.4)]


ZONAS = {
    "cuadriceps": zona_cuadriceps,
    "isquiotibiales": zona_isquiotibiales,
    "gluteos": zona_gluteos,
    "aductores": zona_aductores,
    "gemelos": zona_gemelos,
    "erectores": zona_erectores,
    "abdominales": zona_abdominales,
}


# --------------------------------------------------------------------------- #
#  Pintado
# --------------------------------------------------------------------------- #

def _fuente(tam, negrita=False):
    nombres = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Helvetica.ttc"] if negrita
        else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc"]
    )
    for n in nombres:
        try:
            return ImageFont.truetype(n, tam)
        except OSError:
            pass
    return ImageFont.load_default(size=tam)


def _mascara(pose, poligonos, suavizado):
    capa = Image.new("L", (pose.w, pose.h), 0)
    dib = ImageDraw.Draw(capa)
    for poli in poligonos:
        dib.polygon([(float(x), float(y)) for x, y in poli], fill=255)
    capa = capa.filter(ImageFilter.GaussianBlur(suavizado))
    return (np.asarray(capa, dtype=np.float32) / 255.0) * pose.silueta


def _insignia(dib, centro, numero, radio, rol):
    x, y = centro
    dib.ellipse([x - radio, y - radio, x + radio, y + radio], fill=COLOR_ROL[rol],
                outline=(255, 255, 255), width=max(2, radio // 6))
    f = _fuente(int(radio * 1.15), negrita=True)
    dib.text((x, y), str(numero), font=f, fill=(255, 255, 255), anchor="mm")


def _ajustar(dib, texto, tam, ancho_max, negrita=False):
    """Fuente del mayor tamaño (<= tam) con la que `texto` cabe en `ancho_max` px."""
    while tam > 8:
        f = _fuente(tam, negrita)
        if dib.textlength(texto, font=f) <= ancho_max:
            return f
        tam -= 1
    return _fuente(tam, negrita)


def _leyenda(foto, titulo, filas, vista):
    h = foto.height
    ancho = max(380, int(h * 0.42))
    m = int(ancho * 0.07)
    util = ancho - 2 * m
    panel = Image.new("RGB", (ancho, h), (18, 18, 22))
    dib = ImageDraw.Draw(panel)

    y = m
    f = _ajustar(dib, titulo.upper(), int(ancho * 0.075), util, negrita=True)
    dib.text((m, y), titulo.upper(), font=f, fill=(255, 255, 255))
    y += int(ancho * 0.10)
    sub = f"Músculos implicados · vista {vista}"
    dib.text((m, y), sub, font=_ajustar(dib, sub, int(ancho * 0.045), util), fill=(170, 170, 180))
    y += int(ancho * 0.11)

    radio = int(ancho * 0.042)
    x = m + radio * 2 + int(ancho * 0.04)
    paso = int(ancho * 0.15)
    for numero, musculo, visible in filas:
        _insignia(dib, (m + radio, y + radio), numero, radio, musculo.rol)
        f = _ajustar(dib, musculo.nombre, int(ancho * 0.055), ancho - m - x, negrita=True)
        dib.text((x, y - int(radio * 0.25)), musculo.nombre, font=f, fill=(255, 255, 255))
        rol = ROL_TEXTO[musculo.rol] + ("" if visible else " · no visible")
        color = (255, 120, 120) if musculo.rol == "principal" else (200, 200, 205)
        if not visible:
            color = (130, 130, 140)
        dib.text((x, y + int(radio * 1.2)), rol, font=_ajustar(dib, rol, int(ancho * 0.040), ancho - m - x),
                 fill=color)
        y += paso

    # Escala de intensidad
    f_rol = _fuente(int(ancho * 0.040))
    y = h - m - int(ancho * 0.16)
    for rol in ("principal", "secundario"):
        dib.rounded_rectangle([m, y, m + int(ancho * 0.12), y + int(ancho * 0.05)], radius=6, fill=COLOR_ROL[rol])
        dib.text((m + int(ancho * 0.16), y + int(ancho * 0.025)), ROL_TEXTO[rol], font=f_rol,
                 fill=(220, 220, 225), anchor="lm")
        y += int(ancho * 0.08)

    lienzo = Image.new("RGB", (foto.width + ancho, h))
    lienzo.paste(foto, (0, 0))
    lienzo.paste(panel, (foto.width, 0))
    return lienzo


def _separar(puntos, distancia, ancho, alto, iteraciones=60):
    """Empuja las insignias que se solapan hasta que queden a `distancia` px entre sí."""
    p = np.array(puntos, dtype=np.float32)
    for _ in range(iteraciones):
        movido = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = p[j] - p[i]
                dist = float(np.linalg.norm(d))
                if dist < distancia:
                    u = d / dist if dist > 1e-3 else np.array([0.0, 1.0])
                    empuje = (distancia - dist) / 2 + 0.5
                    p[i] -= u * empuje
                    p[j] += u * empuje
                    movido = True
        if not movido:
            break
    p[:, 0] = np.clip(p[:, 0], distancia / 2, ancho - distancia / 2)
    p[:, 1] = np.clip(p[:, 1], distancia / 2, alto - distancia / 2)
    return [tuple(map(int, q)) for q in p]


def marcar_musculos(img: Image.Image, ejercicio: str, leyenda: bool = True) -> Image.Image:
    """Devuelve una copia de `img` con los músculos del ejercicio marcados en rojo."""
    if ejercicio not in MAPAS:
        raise KeyError(f"no hay mapa muscular para '{ejercicio}'. Disponibles: {list(MAPAS)}")
    mapa = MAPAS[ejercicio]
    img = img.convert("RGB")
    pose = Pose(img)

    base = np.asarray(img, dtype=np.float32) / 255.0
    luz = (base @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    # Rojo que conserva el sombreado de la foto (se sigue viendo la forma del músculo)
    # y que no se apaga sobre ropa oscura
    rojo = np.array(ROJO, dtype=np.float32) / 255.0
    tinte = np.clip(rojo * (0.70 + 0.45 * luz) + 0.08 * luz, 0, 1)

    suavizado = max(2.0, 0.006 * math.hypot(pose.w, pose.h))
    alfa = np.zeros((pose.h, pose.w), dtype=np.float32)
    marcados = []
    for numero, musculo in enumerate(mapa["musculos"], 1):
        m = _mascara(pose, ZONAS[musculo.zona](pose), suavizado)
        visible = m.max() > 0.25
        if visible:
            alfa = np.maximum(alfa, m * ALFA[musculo.rol])
        marcados.append((numero, musculo, visible, m))

    salida = base * (1 - alfa[..., None]) + tinte * alfa[..., None]
    foto = Image.fromarray((salida * 255).round().astype(np.uint8))

    # Número de cada músculo sobre su zona (sin que se tapen entre sí)
    dib = ImageDraw.Draw(foto)
    radio = max(9, int(0.018 * pose.h))
    visibles = [(n, mu, m) for n, mu, v, m in marcados if v]
    centros = []
    for _, _, m in visibles:
        ys, xs = np.nonzero(m > 0.5 * m.max())
        centros.append((xs.mean(), ys.mean()))
    for (numero, musculo, _), c in zip(visibles, _separar(centros, radio * 2.3, pose.w, pose.h)):
        _insignia(dib, c, numero, radio, musculo.rol)

    if not leyenda:
        return foto
    filas = [(n, mu, v) for n, mu, v, _ in marcados]
    return _leyenda(foto, mapa["titulo"], filas, pose.vista)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Marca en rojo los músculos de un ejercicio en una foto.")
    p.add_argument("imagen")
    p.add_argument("--ejercicio", default="sentadilla", choices=list(MAPAS))
    p.add_argument("--salida", help="Ruta de salida (por defecto <imagen>_musculos.png)")
    p.add_argument("--sin-leyenda", action="store_true")
    a = p.parse_args()

    origen = Path(a.imagen)
    destino = Path(a.salida) if a.salida else origen.with_name(f"{origen.stem}_musculos.png")
    marcar_musculos(Image.open(origen), a.ejercicio, leyenda=not a.sin_leyenda).save(destino)
    print(f"guardada: {destino}")
