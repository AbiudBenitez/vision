"""Preprocesamiento: grises, blur, threshold y morfología.

Es el paso 1 del proyecto y la base de todos los detectores. Antes de escribir
cualquier detector hay que calibrar la iluminación mirando el resultado de
`binarizar` en vivo: si el borde de la funda no queda limpio y cerrado aquí,
ningún detector aguas abajo será confiable.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ParamsPreproceso:
    # Kernel impar de Gauss. Suaviza ruido del sensor antes de umbralizar para
    # que Otsu no persiga píxeles sueltos. Impar por requisito de OpenCV.
    blur_ksize: int = 5
    # Otsu elige el umbral automáticamente; sirve porque la iluminación es fija
    # dentro de una sesión pero varía entre sesiones de captura, así no hay que
    # recalibrar un número mágico cada vez.
    usar_otsu: bool = True
    # Umbral manual de respaldo cuando usar_otsu=False (p.ej. si el fondo y la
    # funda quedan con histograma no bimodal y Otsu falla).
    umbral_manual: int = 127
    # La funda es el objeto claro sobre fondo... o al revés. invertir=True deja
    # la funda en blanco (255) para que findContours la tome como primer plano.
    invertir: bool = True
    # Cierre morfológico: tapa huecos finos del borde binarizado sin engordar
    # la silueta. Tamaño del elemento estructurante (impar).
    cierre_ksize: int = 5
    # Iteraciones del cierre. Más iteraciones = tolera bordes más rotos, pero
    # empieza a fusionar los barrenos con el exterior. Subir con cuidado.
    cierre_iter: int = 1


def a_grises(frame: np.ndarray) -> np.ndarray:
    """BGR -> gris. Todo el pipeline clásico trabaja en un solo canal."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def binarizar(frame: np.ndarray, params: ParamsPreproceso) -> tuple[np.ndarray, dict]:
    """Devuelve (binaria, resultados) lista para contornos/Hough.

    resultados incluye el umbral efectivo que usó Otsu, útil para mostrarlo en
    pantalla durante la calibración de iluminación.
    """
    gris = a_grises(frame)

    k = params.blur_ksize | 1  # fuerza impar
    suave = cv2.GaussianBlur(gris, (k, k), 0)

    tipo = cv2.THRESH_BINARY_INV if params.invertir else cv2.THRESH_BINARY
    if params.usar_otsu:
        umbral_usado, binaria = cv2.threshold(
            suave, 0, 255, tipo | cv2.THRESH_OTSU
        )
    else:
        umbral_usado, binaria = cv2.threshold(
            suave, params.umbral_manual, 255, tipo
        )

    # Cierre = dilatación seguida de erosión. Reconecta el contorno de la funda
    # si la binarización lo dejó con micro-cortes, sin desplazar los bordes.
    kc = params.cierre_ksize | 1
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    binaria = cv2.morphologyEx(
        binaria, cv2.MORPH_CLOSE, elem, iterations=params.cierre_iter
    )

    return binaria, {"umbral_usado": float(umbral_usado)}


def detectar_preproceso(
    frame: np.ndarray, params: ParamsPreproceso
) -> tuple[np.ndarray, dict]:
    """Firma estándar. En modo calibración muestra la binaria en BGR para que
    main.py la dibuje igual que cualquier otro detector."""
    binaria, res = binarizar(frame, params)
    anotado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        anotado,
        f"umbral={res['umbral_usado']:.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return anotado, res
