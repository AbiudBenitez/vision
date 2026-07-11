"""Detección de agujeros vía jerarquía de contornos (paso 2, base del resto).

Tema del temario: `findContours` con `RETR_CCOMP`. CCOMP organiza los contornos
en dos niveles: exteriores (la silueta de la funda) e interiores (los barrenos,
que son huecos DENTRO de la silueta). Contar los hijos del contorno de la funda
nos dice cuántos barrenos hay, sin confundirlos con manchas del fondo.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import ParamsPreproceso, binarizar


@dataclass
class ParamsHoles:
    pre: ParamsPreproceso = None
    # Área mínima (px^2) para que un contorno exterior cuente como "la funda" y
    # no como ruido. Depende de la distancia cámara-pieza; calibrar contra
    # data/. Valor por defecto pensado para funda que llena ~1/3 del frame.
    area_min_funda: float = 15000.0
    # Área mínima de un hueco interior para contarlo como barreno real y
    # descartar poros/artefactos de binarización.
    area_min_barreno: float = 80.0
    # Barrenos esperados en una funda buena (p.ej. 1 cámara + 1 flash). El
    # veredicto compara el conteo real contra esto.
    barrenos_esperados: int = 2


def _contorno_funda(contornos, jerarquia, area_min):
    """Regresa el índice del contorno exterior más grande = la funda.

    En RETR_CCOMP los exteriores tienen jerarquia[i][3] == -1 (sin padre).
    Elegimos el de mayor área para ignorar restos del fondo.
    """
    mejor_i, mejor_area = -1, 0.0
    for i, c in enumerate(contornos):
        if jerarquia[0][i][3] != -1:  # tiene padre => es un hueco, no la funda
            continue
        area = cv2.contourArea(c)
        if area > mejor_area and area >= area_min:
            mejor_i, mejor_area = i, area
    return mejor_i


def detectar_holes(
    frame: np.ndarray, params: ParamsHoles, binaria: np.ndarray = None
) -> tuple[np.ndarray, dict]:
    """Cuenta barrenos como huecos internos del contorno de la funda.

    `binaria` opcional: si el llamador ya binarizó (inspección completa lo hace
    una sola vez para los tres detectores), se reusa en vez de repetir el
    preprocesamiento por cada detector.
    """
    if binaria is None:
        pre = params.pre or ParamsPreproceso()
        binaria, _ = binarizar(frame, pre)

    # RETR_CCOMP + jerarquía: necesitamos saber quién es hijo de quién para
    # distinguir barrenos (hijos de la funda) de objetos sueltos del fondo.
    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    anotado = frame.copy()
    if jerarquia is None or len(contornos) == 0:
        return anotado, {"barrenos": 0, "ok": False, "motivo": "sin_contornos"}

    i_funda = _contorno_funda(contornos, jerarquia, params.area_min_funda)
    if i_funda == -1:
        return anotado, {"barrenos": 0, "ok": False, "motivo": "sin_funda"}

    cv2.drawContours(anotado, contornos, i_funda, (255, 0, 0), 2)

    # Los barrenos son los hijos directos del contorno de la funda: recorremos
    # la cadena de hermanos que cuelgan de i_funda (jerarquia[i][2] = primer
    # hijo, jerarquia[i][0] = siguiente hermano).
    barrenos = []
    hijo = jerarquia[0][i_funda][2]
    while hijo != -1:
        area = cv2.contourArea(contornos[hijo])
        if area >= params.area_min_barreno:
            barrenos.append(hijo)
            cv2.drawContours(anotado, contornos, hijo, (0, 0, 255), 2)
        hijo = jerarquia[0][hijo][0]

    n = len(barrenos)
    ok = n == params.barrenos_esperados
    cv2.putText(
        anotado,
        f"barrenos={n}/{params.barrenos_esperados} {'OK' if ok else 'NG'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if ok else (0, 0, 255),
        2,
    )
    return anotado, {
        "barrenos": n,
        "esperados": params.barrenos_esperados,
        "ok": ok,
        "indices_barrenos": barrenos,
        "indice_funda": i_funda,
    }
