"""Loop de webcam + menú de modos para inspección de fundas.

main.py NO contiene lógica de visión: solo captura frames, despacha al módulo
del modo activo y dibuja el resultado. Cada detector expone la firma común
`detectar_x(frame, params) -> (frame_anotado, resultados)`, así que cambiar de
modo es cambiar la función y el dict de params, nada más. Los params viven en
UN solo objeto `ParamsInspeccion` (src/config.py) compartido por todos los
modos y por el veredicto, para que calibrar una vez valga para todo.

Teclas:
  1 preprocess (calibrar iluminación)   5 ellipses (deformación barreno)
  2 holes (contar barrenos)             6 shapes (polígono/esquinas)
  3 circulos (Hough)                    7 motion (trigger)
  4 lineas (Hough)                      8 inspección completa (veredicto OK/NG)
  9 describir (qué funda veo)           i inspeccionar frame actual
  q salir

El modo 9 no juzga: reporta la silueta y los barrenos que encuentre, sin esperar
un número concreto. Es el que sirve con una funda distinta a la calibrada.

Uso:
  python main.py                # webcam (default cam 0)
  python main.py --cam 1        # otra webcam
  python main.py --img data/x.jpg   # una foto fija en vez de webcam
"""
import argparse
import sys

import cv2

from src.config import ParamsInspeccion
from src.preprocess import detectar_preproceso, escalar_a_ancho
from src.holes import detectar_holes
from src.hough import detectar_circulos, detectar_lineas
from src.ellipses import detectar_elipses
from src.shapes import detectar_shapes
from src.motion import TriggerInspeccion
from src.inspeccion import inspeccion_completa
from src.reporte import describir

MODO_MOTION = ord("7")
MODO_COMPLETA = ord("8")
MODO_DESCRIBIR = ord("9")


def _cerrar_ventana(nombre: str) -> None:
    """Cierra una ventana si existe. destroyWindow lanza en algunos backends si
    la ventana no está abierta, por eso se envuelve."""
    try:
        cv2.destroyWindow(nombre)
    except cv2.error:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="índice de webcam")
    ap.add_argument("--img", type=str, default=None, help="ruta de foto fija")
    args = ap.parse_args()

    fuente_img = None
    cap = None
    if args.img:
        fuente_img = cv2.imread(args.img)
        if fuente_img is None:
            print(f"No se pudo leer la imagen: {args.img}", file=sys.stderr)
            return 1
    else:
        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            print(f"No se pudo abrir la webcam {args.cam}", file=sys.stderr)
            return 1

    cfg = ParamsInspeccion()  # fuente única de params, compartida por todo

    # Cada modo con detector simple = (nombre, funcion, params). motion y
    # completa se manejan aparte porque llevan estado / combinan detectores.
    modos = {
        ord("1"): ("preprocess", detectar_preproceso, cfg.pre),
        ord("2"): ("holes", detectar_holes, cfg.holes),
        ord("3"): ("circulos", detectar_circulos, cfg.circulos),
        ord("4"): ("lineas", detectar_lineas, cfg.lineas),
        ord("5"): ("ellipses", detectar_elipses, cfg.ellipses),
        ord("6"): ("shapes", detectar_shapes, cfg.shapes),
    }

    modo_actual = ord("1")  # arranca en preprocess para calibrar iluminación
    trigger = TriggerInspeccion(cfg.motion)
    ultimo_veredicto = None  # frame congelado tras un trigger

    print(__doc__)
    while True:
        if fuente_img is not None:
            frame = fuente_img.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break

        # Normalizar la escala ANTES de cualquier detector: los params están
        # calibrados a cfg.ancho_trabajo. Sin esto, una foto de celular (4000px)
        # y la webcam (640px) recorren caminos distintos con los mismos números.
        frame = escalar_a_ancho(frame, cfg.ancho_trabajo)

        if modo_actual == MODO_MOTION:
            anotado, res = trigger.procesar(frame)
            if res["disparar"]:
                # Trigger: corre la inspección y congela el veredicto para verlo.
                ultimo_veredicto, _ = inspeccion_completa(frame, cfg)
            if ultimo_veredicto is not None:
                cv2.imshow("veredicto", ultimo_veredicto)
        elif modo_actual == MODO_COMPLETA:
            anotado, _ = inspeccion_completa(frame, cfg)
        elif modo_actual == MODO_DESCRIBIR:
            anotado, _ = describir(frame, cfg)
        else:
            _, fn, params = modos[modo_actual]
            anotado, _ = fn(frame, params)

        cv2.imshow("inspeccion", anotado)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("i"):
            # Inspección manual a demanda sobre el frame actual.
            ultimo_veredicto, _ = inspeccion_completa(frame, cfg)
            cv2.imshow("veredicto", ultimo_veredicto)
        elif k == MODO_MOTION:
            modo_actual = MODO_MOTION
        elif k == MODO_COMPLETA:
            modo_actual = MODO_COMPLETA
        elif k == MODO_DESCRIBIR:
            modo_actual = MODO_DESCRIBIR
            _cerrar_ventana("veredicto")
            ultimo_veredicto = None
        elif k in modos:
            modo_actual = k
            # Al salir de motion la ventana de veredicto queda huérfana; ciérrala.
            _cerrar_ventana("veredicto")
            ultimo_veredicto = None

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
