"""Modo descriptivo: NO juzga la funda, la DESCRIBE.

Por qué existe, separado del veredicto:

`inspeccion.py` responde "¿esta funda está bien?", y para eso necesita saber de
antemano qué esperar (3 barrenos, 4 vértices). Eso solo funciona con la funda
para la que se calibró: al poner otra pieza, con otro número de barrenos, el
veredicto la rechaza por "faltan barrenos" cuando en realidad está sana.

Este modo responde otra pregunta: "¿QUÉ estoy viendo?". Reporta la silueta y los
agujeros que encuentra, sin compararlos contra ningún número esperado, así que
funciona con una funda que nunca se ha visto antes. Es también el modo honesto
para condiciones de luz imperfectas: en vez de emitir un OK/NG en el que no se
puede confiar, dice lo que midió y advierte cuando la medición es dudosa.

Los detectores son los mismos (jerarquía de contornos, approxPolyDP, fitEllipse,
Hough): lo que cambia es que aquí ninguno tiene derecho a reprobar a la pieza.
"""
import cv2
import numpy as np

from .config import ParamsInspeccion
from .preprocess import binarizar
from .holes import (
    _circularidad,
    _contorno_funda,
    area_referencia,
    indices_barrenos,
    rectangularidad,
)

# Por debajo de esta rectangularidad, la silueta está mordida (por un reflejo o
# por el estampado alcanzando el borde de la pieza) y el conteo de vértices deja
# de significar nada. Medido: fundas bien segmentadas dan 0.90-0.96; un frame de
# webcam con un reflejo que le come media silueta da 0.52. 0.85 los separa.
RECT_MIN_CONFIABLE = 0.85


def describir(frame: np.ndarray, cfg: ParamsInspeccion) -> tuple[np.ndarray, dict]:
    """Describe la funda del frame: silueta y barrenos. Nunca emite OK/NG.

    Devuelve (frame_anotado, datos) con la misma firma que un detector, para que
    main.py lo despache igual que cualquier otro modo.
    """
    binaria, _ = binarizar(frame, cfg.pre)
    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    anotado = frame.copy()
    if jerarquia is None or len(contornos) == 0:
        return anotado, {"hay_funda": False, "motivo": "sin_contornos"}

    area_frame = float(binaria.shape[0] * binaria.shape[1])
    i_funda = _contorno_funda(
        contornos, jerarquia, cfg.holes.frac_min_funda * area_frame, binaria.shape
    )
    if i_funda == -1:
        cv2.putText(
            anotado, "sin funda en el encuadre", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return anotado, {"hay_funda": False, "motivo": "sin_funda"}

    c_funda = contornos[i_funda]
    a_ref = area_referencia(c_funda)
    rect = rectangularidad(c_funda)

    # Silueta y polígono: se dibuja el contorno real, no el rectángulo, para que
    # se vea en pantalla si la segmentación se comió un pedazo.
    cv2.drawContours(anotado, [c_funda], -1, (255, 0, 0), 2)
    eps = cfg.shapes.epsilon_frac * cv2.arcLength(c_funda, True)
    poligono = cv2.approxPolyDP(c_funda, eps, True)
    for p in poligono.reshape(-1, 2):
        cv2.circle(anotado, tuple(int(v) for v in p), 5, (0, 255, 255), -1)

    # Barrenos: se reportan TODOS los que se encuentren, sin esperar un número.
    barrenos = []
    for i in indices_barrenos(contornos, jerarquia, i_funda, cfg.holes):
        c = contornos[i]
        if len(c) < 5:
            continue
        (cx, cy), (e1, e2), _ = cv2.fitEllipse(c)
        menor, mayor = sorted((e1, e2))
        barrenos.append({
            "centro": (float(cx), float(cy)),
            "radio_px": float(0.5 * mayor),
            # Diámetro relativo al tamaño de la pieza: es lo único comparable
            # entre fundas distintas y entre montajes distintos.
            "diam_rel": float(mayor / np.sqrt(a_ref)) if a_ref > 0 else 0.0,
            "ratio_ejes": float(menor / mayor) if mayor > 0 else 0.0,
            "circularidad": float(_circularidad(c)),
        })
        cv2.drawContours(anotado, contornos, i, (0, 0, 255), 2)

    barrenos.sort(key=lambda b: b["centro"][1])  # de arriba a abajo, estable
    datos = {
        "hay_funda": True,
        "vertices": len(poligono),
        "rectangularidad": rect,
        "silueta_confiable": rect >= RECT_MIN_CONFIABLE,
        "barrenos": len(barrenos),
        "detalle_barrenos": barrenos,
    }

    y = 30
    cv2.putText(
        anotado, f"barrenos: {len(barrenos)}", (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    y += 30
    cv2.putText(
        anotado, f"vertices: {len(poligono)}", (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    y += 30
    if not datos["silueta_confiable"]:
        # Avisar en pantalla en vez de callar: con la silueta mordida, el número
        # de vértices de arriba no significa nada y el operador debe saberlo.
        cv2.putText(
            anotado, f"silueta dudosa (rect={rect:.2f}): mueve la luz/pieza",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
        )
        y += 26
    for i, b in enumerate(barrenos):
        cv2.putText(
            anotado,
            f"#{i}: d={b['diam_rel']:.3f} circ={b['circularidad']:.2f}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        y += 20

    return anotado, datos
