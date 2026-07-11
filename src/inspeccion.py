"""Modo integración: corre todos los detectores y emite un veredicto único.

Es el modo de demostración final. Vive fuera de main.py porque main.py no debe
contener lógica de visión: aquí se combinan conteo de barrenos, circularidad,
forma del contorno, diámetro (Hough círculos) y ortogonalidad de bordes (Hough
líneas) en un OK/NG global.
"""
import cv2
import numpy as np

from .preprocess import binarizar
from .holes import detectar_holes
from .hough import detectar_circulos, detectar_lineas
from .ellipses import detectar_elipses
from .shapes import detectar_shapes
from .config import ParamsInspeccion


def _bordes_ortogonales(lineas, tol_grados: float = 15.0) -> bool:
    """True si entre los segmentos de Hough hay al menos uno ~horizontal y uno
    ~vertical: los cuatro bordes de una funda sana forman dos pares ortogonales.

    Criterio GRUESO y frágil a propósito: no verifica paralelismo ni que sean
    exactamente cuatro bordes, solo presencia de ambas orientaciones. Un fondo
    con líneas o una funda girada lo engañan. Sirve como primer filtro; se
    endurece cuando haya fotos en data/ para calibrar tolerancia.
    """
    horiz = vert = False
    for l in lineas:
        # Orientación de la recta plegada a 0..90: 0=horizontal, 90=vertical.
        a = abs(l["ang"]) % 180
        if a > 90:
            a = 180 - a
        if a <= tol_grados:
            horiz = True
        elif a >= 90 - tol_grados:
            vert = True
    return horiz and vert


def inspeccion_completa(
    frame: np.ndarray, cfg: ParamsInspeccion
) -> tuple[np.ndarray, dict]:
    """Corre todos los detectores geométricos y devuelve (anotado, veredicto).

    Binariza UNA sola vez y comparte la binaria con los detectores basados en
    contornos (holes/shapes/ellipses); Hough (círculos/líneas) trabaja sobre
    gris, así que corre aparte. Misma firma que un detector para que main.py lo
    despache igual.
    """
    binaria, _ = binarizar(frame, cfg.pre)

    _, r_holes = detectar_holes(frame, cfg.holes, binaria)
    anotado, r_shapes = detectar_shapes(frame, cfg.shapes, binaria)
    _, r_elip = detectar_elipses(frame, cfg.ellipses, binaria)
    _, r_circ = detectar_circulos(frame, cfg.circulos)
    _, r_lin = detectar_lineas(frame, cfg.lineas)

    n_esperados = cfg.holes.barrenos_esperados
    circulos = r_circ["circulos"]
    # Diámetro OK: se detectan al menos los barrenos esperados y todos los
    # radios caen dentro del rango calibrado (fuera de rango = barreno sobre/
    # sub-dimensionado).
    radios_en_rango = all(
        cfg.circulos.radio_min <= c["r"] <= cfg.circulos.radio_max
        for c in circulos
    )
    diametro_ok = len(circulos) >= n_esperados and radios_en_rango

    checks = {
        "barrenos_ok": r_holes.get("ok", False),
        "forma_ok": r_shapes.get("ok", False),
        "circulares_ok": r_elip.get("todos_circulares", False),
        "diametro_ok": diametro_ok,
        "bordes_ok": _bordes_ortogonales(r_lin["lineas"]),
    }
    veredicto_ok = all(checks.values())

    y = 30
    for nombre, ok in checks.items():
        cv2.putText(
            anotado, f"{nombre}: {'OK' if ok else 'NG'}", (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (0, 0, 255), 2,
        )
        y += 28
    cv2.putText(
        anotado, f"VEREDICTO: {'ACEPTADA' if veredicto_ok else 'RECHAZADA'}",
        (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 255, 0) if veredicto_ok else (0, 0, 255), 2,
    )
    return anotado, {"ok": veredicto_ok, "checks": checks}
