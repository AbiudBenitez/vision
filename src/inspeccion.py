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


def _diametros_ok(barrenos, circulos, n_esperados: int, tolerancia: float) -> bool:
    """True si CADA barreno esperado tiene un círculo de Hough encima.

    No basta con contar los círculos que Hough devuelve en todo el frame: el
    estampado de la funda genera 0-1 falsos positivos por foto, y ese ruido puede
    completar el conteo de una funda a la que le falta un barreno (en data/,
    def_tapado da exactamente eso: 2 barrenos reales + 1 falso = 3). El conteo
    crudo diría "diámetro OK" sobre una pieza defectuosa.

    Por eso se EMPAREJA: cada barreno hallado por contornos (holes) debe tener un
    círculo de Hough con el centro encima. Como HoughCircles solo acepta radios
    dentro del rango calibrado, un barreno fuera de diámetro simplemente no
    genera círculo y se queda sin pareja -> reprueba. Así es como se caza el
    barreno chico de def_ovalado (radio 11.8px contra los 15.9-16.9 sanos).

    Los falsos positivos del estampado quedan lejos de cualquier barreno y no
    emparejan con nada, así que dejan de contaminar el veredicto.
    """
    emparejados = 0
    for b in barrenos:
        bx, by = b["centro"]
        if any(
            np.hypot(c["x"] - bx, c["y"] - by) <= tolerancia for c in circulos
        ):
            emparejados += 1
    return emparejados == n_esperados


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
    diametro_ok = _diametros_ok(
        r_elip["barrenos"], r_circ["circulos"], n_esperados,
        tolerancia=0.5 * cfg.circulos.frac_min_dist * frame.shape[1],
    )

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
