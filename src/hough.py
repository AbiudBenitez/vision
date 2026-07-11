"""Transformada de Hough: círculos (barrenos) y líneas (bordes).

Tema del temario: base común `HoughCircles` y `HoughLinesP`. Hough vota en un
espacio de parámetros, así que tolera barrenos parcialmente ocluidos o bordes
con micro-cortes mejor que un ajuste directo de contornos. Aquí sirve para
verificar PRESENCIA y DIÁMETRO de barrenos, y RECTITUD de bordes.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import ParamsPreproceso, a_grises


@dataclass
class ParamsCirculos:
    # dp: resolución inversa del acumulador. 1.2 da margen de tolerancia sin
    # perder precisión; subir si los barrenos no se detectan por ruido.
    dp: float = 1.2
    # Distancia mínima entre centros. Debe ser ~ el diámetro esperado para no
    # detectar dos círculos sobre un mismo barreno. Calibrar contra data/.
    min_dist: float = 40.0
    # param1: umbral alto de Canny interno. param2: votos para aceptar círculo
    # (más bajo = más círculos, más falsos positivos).
    param1: float = 100.0
    param2: float = 30.0
    # Rango de radios esperados (px). Fuera de este rango => barreno fuera de
    # diámetro. Calibrar midiendo un barreno bueno en una foto de data/.
    radio_min: int = 8
    radio_max: int = 40


@dataclass
class ParamsLineas:
    # Umbrales de Canny para extraer bordes antes de Hough. Rango amplio porque
    # la iluminación lateral marca bien el contorno de la funda.
    canny_bajo: int = 50
    canny_alto: int = 150
    # Votos mínimos, largo mínimo de segmento y hueco máximo para unir. Un
    # borde de funda es largo y continuo => umbral de largo alto filtra ruido.
    umbral_votos: int = 60
    largo_min: int = 80
    hueco_max: int = 10


def detectar_circulos(
    frame: np.ndarray, params: ParamsCirculos
) -> tuple[np.ndarray, dict]:
    """HoughCircles sobre gris suavizado. Reporta centro y radio de cada
    barreno detectado; el radio permite juzgar si está fuera de diámetro."""
    gris = a_grises(frame)
    # HoughCircles ya corre Canny internamente; el blur medio evita votos
    # dispersos por textura del plástico.
    gris = cv2.medianBlur(gris, 5)

    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.min_dist,
        param1=params.param1,
        param2=params.param2,
        minRadius=params.radio_min,
        maxRadius=params.radio_max,
    )

    anotado = frame.copy()
    detectados = []
    if circulos is not None:
        for x, y, r in np.uint16(np.around(circulos[0])):
            detectados.append({"x": int(x), "y": int(y), "r": int(r)})
            cv2.circle(anotado, (x, y), r, (0, 255, 0), 2)
            cv2.circle(anotado, (x, y), 2, (0, 0, 255), 3)

    cv2.putText(
        anotado,
        f"circulos={len(detectados)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return anotado, {"circulos": detectados, "n": len(detectados)}


def detectar_lineas(
    frame: np.ndarray, params: ParamsLineas
) -> tuple[np.ndarray, dict]:
    """HoughLinesP sobre bordes de Canny. Cada segmento largo es candidato a
    borde de la funda; su ángulo sirve luego para juzgar paralelismo/ortogonalidad."""
    gris = a_grises(frame)
    bordes = cv2.Canny(gris, params.canny_bajo, params.canny_alto)

    lineas = cv2.HoughLinesP(
        bordes,
        rho=1,
        theta=np.pi / 180,
        threshold=params.umbral_votos,
        minLineLength=params.largo_min,
        maxLineGap=params.hueco_max,
    )

    anotado = frame.copy()
    segmentos = []
    if lineas is not None:
        # HoughLinesP devuelve forma (N,1,4); reshape a (N,4) para desempacar
        # cada segmento de forma robusta entre versiones de OpenCV.
        for x1, y1, x2, y2 in lineas.reshape(-1, 4):
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            segmentos.append({"p1": (int(x1), int(y1)), "p2": (int(x2), int(y2)), "ang": ang})
            cv2.line(anotado, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        anotado,
        f"lineas={len(segmentos)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return anotado, {"lineas": segmentos, "n": len(segmentos)}
