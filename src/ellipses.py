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


@dataclass
class ParamsElipses:
    pre: ParamsPreproceso = None
    # fitEllipse necesita >=5 puntos; además descartamos contornos diminutos
    # que darían ajustes inestables.
    puntos_min: int = 5
    area_min: float = 80.0
    area_max: float = 8000.0
    # Criterio de deformación: ratio = eje_menor / eje_mayor. 1.0 = círculo
    # perfecto. Por debajo de este umbral el barreno se considera ovalado.
    # 0.80 tolera la distorsión de perspectiva leve pero marca óvalos claros;
    # calibrar contra un barreno bueno y uno deformado de data/.
    ratio_min: float = 0.80


def _es_hueco(jerarquia, i) -> bool:
    """En RETR_CCOMP un hueco (barreno) tiene padre (jerarquia[i][3] != -1)."""
    return jerarquia[0][i][3] != -1


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

    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    anotado = frame.copy()
    if jerarquia is None:
        return anotado, {"barrenos": [], "todos_circulares": True}

    barrenos = []
    for i, c in enumerate(contornos):
        if not _es_hueco(jerarquia, i):
            continue
        if len(c) < params.puntos_min:
            continue
        area = cv2.contourArea(c)
        if not (params.area_min <= area <= params.area_max):
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
