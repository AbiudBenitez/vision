"""Transformada de Hough: círculos (barrenos) y líneas (bordes).

Tema del temario: base común `HoughCircles` y `HoughLinesP`. Hough vota en un
espacio de parámetros, así que tolera barrenos parcialmente ocluidos o bordes
con micro-cortes mejor que un ajuste directo de contornos. Aquí sirve para
verificar PRESENCIA y DIÁMETRO de barrenos, y RECTITUD de bordes.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import ParamsPreproceso, a_grises, binarizar
from .holes import ParamsHoles, _contorno_funda


@dataclass
class ParamsCirculos:
    # Radios y distancias van como fracción de sqrt(ÁREA DE LA FUNDA), no en px
    # ni como fracción del ancho del frame. La razón importa:
    #
    # Un barreno se ve más grande cuando la funda se ve más grande, y eso no
    # depende del encuadre sino de la pieza. Normalizar por el ancho del frame
    # PARECE funcionar hasta que cambia el aspect ratio: las fotos de data/ son
    # verticales (640x853) y la webcam entrega horizontal (640x360). Mismo ancho,
    # área muy distinta. Medido: los barrenos sanos dan r/ancho = 0.0249-0.0264
    # en las fotos pero 0.0189-0.0202 en la webcam, y el barreno DEFECTUOSO de
    # def_ovalado da 0.0185 -> un barreno sano visto por webcam es indistinguible
    # de uno defectuoso visto en foto. Ese criterio rechaza piezas buenas en vivo.
    #
    # Con r/sqrt(area_funda) los dos montajes coinciden: sanos 0.0436-0.0469 en
    # fotos y 0.0453-0.0484 en webcam, contra 0.0328 del defectuoso. Es la
    # magnitud correcta porque sqrt(area) tiene unidades de longitud, igual que
    # el radio, así que el cociente es adimensional e invariante a la escala.

    # Solo para el modo suelto de círculos, que debe deducir el tamaño de la
    # funda por su cuenta. En la inspección completa el área viene dada por holes.
    pre: ParamsPreproceso = None
    holes: ParamsHoles = None

    # dp: resolución inversa del acumulador. 1.2 da margen de tolerancia sin
    # perder precisión; subir si los barrenos no se detectan por ruido.
    dp: float = 1.2
    # Distancia mínima entre centros, para no sacar dos círculos del mismo
    # barreno. Los barrenos están separados ~48px y su diámetro es ~33px (medido
    # en data/, donde sqrt(area_funda)=365): 0.10 cae entre ambos, así que no
    # fusiona barrenos vecinos ni duplica uno solo.
    frac_min_dist: float = 0.10
    # param1: umbral alto de Canny interno. param2: votos para aceptar círculo.
    # param2=30 medido contra las 5 fotos: da los 3 barrenos con 0-1 falsos. Con
    # 20 entran 4-8 falsos del estampado; con 40 se PIERDE un barreno sano en
    # buena_02 (falso NG), que es el error más caro. 30 es el punto dulce.
    param1: float = 100.0
    param2: float = 30.0
    # Rango de radios aceptados. Este rango ES el criterio de "barreno dentro de
    # diámetro": Hough sencillamente no reporta lo que cae fuera, así que un
    # barreno fuera de tolerancia se queda sin círculo y reprueba.
    # Sanos medidos (fotos + webcam): 0.0436-0.0484. Defectuoso: 0.0328.
    # El piso 0.038 se planta en el hueco entre ambos grupos; el techo 0.056 deja
    # ~1.15x de holgura sobre el barreno sano más grande visto.
    frac_radio_min: float = 0.038
    frac_radio_max: float = 0.056

    def radios_px(self, area_funda: float) -> tuple[int, int]:
        """Rango de radios en píxeles para el tamaño de funda detectado.

        round() y no int(): truncar rebaja el piso casi un píxel entero, y el
        margen que separa "fuera de diámetro" de "aceptado por error" se mide
        justamente en píxeles sueltos.
        """
        escala = math.sqrt(max(area_funda, 1.0))
        return (
            max(1, round(self.frac_radio_min * escala)),
            max(2, round(self.frac_radio_max * escala)),
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


def area_funda_de(frame: np.ndarray, pre: ParamsPreproceso, holes: ParamsHoles) -> float:
    """Área en px del contorno de la funda. 0.0 si no hay funda en el frame.

    Hough necesita el tamaño de la funda para saber qué radio ESPERAR (ver
    ParamsCirculos). En la inspección completa ese dato ya lo calculó holes y se
    pasa hecho; esto es solo para el modo suelto de círculos, que no tiene a
    nadie que se lo dé.
    """
    binaria, _ = binarizar(frame, pre)
    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if jerarquia is None or len(contornos) == 0:
        return 0.0
    area_frame = float(binaria.shape[0] * binaria.shape[1])
    i = _contorno_funda(
        contornos, jerarquia, holes.frac_min_funda * area_frame, binaria.shape
    )
    return cv2.contourArea(contornos[i]) if i != -1 else 0.0


def detectar_circulos(
    frame: np.ndarray, params: ParamsCirculos, area_funda: float = None
) -> tuple[np.ndarray, dict]:
    """HoughCircles sobre gris suavizado. Reporta centro y radio de cada
    barreno detectado; el radio permite juzgar si está fuera de diámetro.

    `area_funda` fija la escala del radio esperado. Si no se pasa (modo suelto),
    se deduce del frame. Sin funda no hay escala de referencia y no tiene sentido
    buscar barrenos: se devuelve vacío en vez de inventar un rango de radios.
    """
    if area_funda is None:
        area_funda = area_funda_de(
            frame, params.pre or ParamsPreproceso(), params.holes or ParamsHoles()
        )

    anotado_vacio = frame.copy()
    if area_funda <= 0:
        cv2.putText(
            anotado_vacio, "sin funda", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return anotado_vacio, {"circulos": [], "n": 0}

    gris = a_grises(frame)
    # HoughCircles ya corre Canny internamente; el blur medio evita votos
    # dispersos por textura del plástico.
    gris = cv2.medianBlur(gris, 5)

    escala = math.sqrt(area_funda)
    radio_min, radio_max = params.radios_px(area_funda)

    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.frac_min_dist * escala,
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
