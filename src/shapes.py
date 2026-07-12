"""Polígonos y esquinas: `approxPolyDP` + `goodFeaturesToTrack`.

Tema del temario: aproximación poligonal y detección de esquinas. Una funda
sana, vista de frente, aproxima a un rectángulo redondeado => approxPolyDP
sobre su contorno debe dar ~4 vértices. Un número distinto delata una esquina
recortada/golpeada o un borde deformado. goodFeaturesToTrack (Shi-Tomasi)
localiza las esquinas reales para inspeccionarlas puntualmente.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import ParamsPreproceso, binarizar
from .holes import _contorno_funda


@dataclass
class ParamsShapes:
    pre: ParamsPreproceso = None
    # Fracción del frame, no px^2: mismo motivo que en ParamsHoles (las fotos de
    # data/ y la webcam difieren ~40x en área). Debe coincidir con
    # ParamsHoles.frac_min_funda para que ambos detectores elijan LA MISMA funda.
    frac_min_funda: float = 0.02
    # epsilon de approxPolyDP como fracción del perímetro. 0.02 es el valor
    # típico para simplificar a un polígono limpio sin colapsar esquinas
    # legítimas; subir si aparecen vértices espurios por ruido del contorno.
    epsilon_frac: float = 0.02
    # Vértices esperados en una funda buena (rectángulo). Distinto => defecto
    # de esquina/borde. El chaflán de esquinas redondeadas puede sumar vértices;
    # calibrar contra data/.
    vertices_esperados: int = 4
    # Parámetros Shi-Tomasi para localizar esquinas dentro de la funda.
    max_esquinas: int = 12
    calidad: float = 0.2
    dist_min: float = 30.0


def detectar_shapes(
    frame: np.ndarray, params: ParamsShapes, binaria: np.ndarray = None
) -> tuple[np.ndarray, dict]:
    """Aproxima el contorno de la funda a un polígono y cuenta sus vértices;
    además marca esquinas Shi-Tomasi para inspección visual.

    `binaria` opcional: reusa la binaria de la inspección completa."""
    if binaria is None:
        pre = params.pre or ParamsPreproceso()
        binaria, _ = binarizar(frame, pre)

    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    anotado = frame.copy()
    if jerarquia is None or len(contornos) == 0:
        return anotado, {"vertices": 0, "ok": False, "motivo": "sin_contornos"}

    area_frame = float(binaria.shape[0] * binaria.shape[1])
    i_funda = _contorno_funda(
        contornos, jerarquia, params.frac_min_funda * area_frame, binaria.shape
    )
    if i_funda == -1:
        return anotado, {"vertices": 0, "ok": False, "motivo": "sin_funda"}

    c = contornos[i_funda]
    eps = params.epsilon_frac * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True)
    n_vert = len(poly)
    ok = n_vert == params.vertices_esperados

    cv2.drawContours(anotado, [poly], -1, (255, 0, 0), 2)
    for v in poly[:, 0]:
        cv2.circle(anotado, tuple(v), 5, (0, 255, 255), -1)

    # Shi-Tomasi sobre la MÁSCARA de la silueta, no sobre el frame en grises.
    # La funda va estampada: sobre el gris, Shi-Tomasi encontraría las esquinas
    # del DIBUJO impreso (que tiene mucho más gradiente que el borde de la
    # pieza) y no las de la funda. La máscara es blanco liso dentro y negro
    # fuera, así que el único gradiente que existe es el contorno físico -> las
    # esquinas que salen son geométricas por construcción, y el estampado
    # simplemente no existe para el detector.
    mascara = np.zeros(binaria.shape, np.uint8)
    cv2.drawContours(mascara, [c], -1, 255, -1)
    esquinas = cv2.goodFeaturesToTrack(
        mascara,
        maxCorners=params.max_esquinas,
        qualityLevel=params.calidad,
        minDistance=params.dist_min,
        mask=mascara,
    )
    pts_esquinas = []
    if esquinas is not None:
        for x, y in np.int32(esquinas[:, 0]):
            pts_esquinas.append((int(x), int(y)))
            cv2.circle(anotado, (x, y), 4, (0, 0, 255), -1)

    cv2.putText(
        anotado,
        f"vertices={n_vert}/{params.vertices_esperados} {'OK' if ok else 'NG'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if ok else (0, 0, 255),
        2,
    )
    return anotado, {
        "vertices": n_vert,
        "esperados": params.vertices_esperados,
        "ok": ok,
        "esquinas": pts_esquinas,
    }
