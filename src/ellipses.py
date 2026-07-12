"""Detección de elipses: `fitEllipse` sobre barrenos para medir deformación.

Tema del temario: ajuste de elipses. Un barreno bueno es circular => la elipse
ajustada tiene ejes casi iguales (ratio ~1). Si el barreno está ovalado o
golpeado, el ratio menor/mayor cae por debajo del umbral y lo marcamos como
deformado. Complementa a Hough: Hough dice "hay círculo", fitEllipse cuantifica
CUÁNTO se aleja de serlo.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import ParamsPreproceso, binarizar
from .holes import ParamsHoles, _contorno_funda, indices_barrenos


@dataclass
class ParamsElipses:
    pre: ParamsPreproceso = None
    # Qué huecos son barrenos NO se decide aquí: se reusa el criterio de
    # holes.py (ver indices_barrenos). Si este módulo eligiera por su cuenta,
    # mediría la deformación de manchas del estampado y el veredicto sería
    # incoherente con el conteo de barrenos.
    holes: ParamsHoles = None
    # fitEllipse necesita >=5 puntos o lanza excepción.
    puntos_min: int = 5
    # Criterio de deformación: ratio = eje_menor / eje_mayor. 1.0 = círculo
    # perfecto.
    #
    # ADVERTENCIA - este es el criterio MÁS FRÁGIL del proyecto. Medido en data/:
    # los barrenos sanos dan 0.80-0.89 y el barreno de def_ovalado da 0.75. El
    # umbral 0.78 los separa, pero el margen es de apenas 0.02 contra el peor
    # barreno bueno: un cambio de iluminación que engorde el borde binarizado
    # basta para cruzarlo en cualquier dirección. NO subir este número hasta que
    # "funcione" con una foto; con más ejemplos de fundas sanas es probable que
    # haya que replantear el criterio.
    #
    # El defecto de def_ovalado se detecta de forma MUCHO más robusta por
    # DIÁMETRO (su radio es 11.8px contra 16.0-16.9px de los sanos, y su área es
    # 2.2x menor -> ver el check diametro_ok en inspeccion.py). Esta elipse es el
    # respaldo que explica POR QUÉ falla, no el juez principal.
    ratio_min: float = 0.78


def detectar_elipses(
    frame: np.ndarray, params: ParamsElipses, binaria: np.ndarray = None
) -> tuple[np.ndarray, dict]:
    """Ajusta una elipse a cada barreno y reporta su ratio de ejes.

    `binaria` opcional: reusa la binaria de la inspección completa para no
    re-binarizar el mismo frame.
    """
    if binaria is None:
        pre = params.pre or ParamsPreproceso()
        binaria, _ = binarizar(frame, pre)
    p_holes = params.holes or ParamsHoles()

    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    anotado = frame.copy()
    if jerarquia is None or len(contornos) == 0:
        return anotado, {"barrenos": [], "todos_circulares": True}

    area_frame = float(binaria.shape[0] * binaria.shape[1])
    i_funda = _contorno_funda(contornos, jerarquia, p_holes.frac_min_funda * area_frame)
    if i_funda == -1:
        return anotado, {"barrenos": [], "todos_circulares": True}

    barrenos = []
    for i in indices_barrenos(contornos, jerarquia, i_funda, p_holes):
        c = contornos[i]
        if len(c) < params.puntos_min:
            continue

        elipse = cv2.fitEllipse(c)  # ((cx,cy),(eje1,eje2),angulo)
        (cx, cy), (e1, e2), ang = elipse
        menor, mayor = sorted((e1, e2))
        ratio = menor / mayor if mayor > 0 else 0.0
        circular = ratio >= params.ratio_min

        barrenos.append(
            {"centro": (cx, cy), "ratio": ratio, "circular": circular, "angulo": ang}
        )
        color = (0, 255, 0) if circular else (0, 0, 255)
        cv2.ellipse(anotado, elipse, color, 2)
        cv2.putText(
            anotado,
            f"{ratio:.2f}",
            (int(cx) - 15, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    todos_ok = all(b["circular"] for b in barrenos) if barrenos else True
    cv2.putText(
        anotado,
        f"barrenos={len(barrenos)} {'OK' if todos_ok else 'DEFORME'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if todos_ok else (0, 0, 255),
        2,
    )
    return anotado, {"barrenos": barrenos, "todos_circulares": todos_ok}
