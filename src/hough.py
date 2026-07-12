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
    # Los radios y distancias van como FRACCIÓN DEL ANCHO del frame, no en px:
    # las fotos de data/ (4000px de ancho) y la webcam de la demo (640px) no
    # comparten escala, así que un radio en px calibrado con una no sirve para la
    # otra. Se convierten a px en tiempo de ejecución. Todos los números medidos
    # abajo están normalizados a un frame de 640px de ancho.

    # dp: resolución inversa del acumulador. 1.2 da margen de tolerancia sin
    # perder precisión; subir si los barrenos no se detectan por ruido.
    dp: float = 1.2
    # Distancia mínima entre centros, para no sacar dos círculos del mismo
    # barreno. Los barrenos están separados ~48px (medido en data/) y su diámetro
    # es ~33px: 35px (0.055*640) cae entre ambos, así que no fusiona barrenos
    # vecinos ni duplica uno solo.
    frac_min_dist: float = 0.055
    # param1: umbral alto de Canny interno. param2: votos para aceptar círculo.
    # param2=30 medido contra las 5 fotos: da los 3 barrenos con 0-1 falsos. Con
    # 20 entran 4-8 falsos del estampado; con 40 se PIERDE un barreno sano en
    # buena_02 (falso NG), que es el error más caro. 30 es el punto dulce.
    param1: float = 100.0
    param2: float = 30.0
    # Rango de radios aceptados. Este rango ES el criterio de "barreno dentro de
    # diámetro": Hough sencillamente no reporta lo que cae fuera.
    # Medido en data/ (a 640px de ancho): los barrenos sanos dan radio 15.9-16.9
    # y el barreno defectuoso de def_ovalado da 11.8. El piso 13px (0.0203*640)
    # se planta entre ambos grupos -> el defectuoso no genera círculo y el check
    # de diámetro lo reprueba. El techo 21px (0.0328*640) deja ~1.25x de holgura
    # sobre el barreno sano más grande.
    frac_radio_min: float = 0.0203
    frac_radio_max: float = 0.0328

    def radios_px(self, ancho: int) -> tuple[int, int]:
        """Rango de radios en píxeles para el ancho de frame actual.

        round() y no int(): truncar movería el piso de 13px a 12px (int(12.99)),
        y el barreno defectuoso de def_ovalado mide 11.8px. Ese píxel de más es
        justo el margen que separa "fuera de diámetro" de "aceptado por error".
        """
        return (
            max(1, round(self.frac_radio_min * ancho)),
            max(2, round(self.frac_radio_max * ancho)),
        )


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

    ancho = frame.shape[1]
    radio_min, radio_max = params.radios_px(ancho)

    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.frac_min_dist * ancho,
        param1=params.param1,
        param2=params.param2,
        minRadius=radio_min,
        maxRadius=radio_max,
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
