"""Detección de movimiento: MOG2 como trigger de inspección.

Tema del temario: sustracción de fondo. La inspección no debe correr en cada
frame: se dispara cuando una funda entra al campo y luego se QUIETA. MOG2
modela el fondo estático; cuando el área de primer plano sube (pieza entrando)
y después vuelve a bajar y se estabiliza, es el momento de inspeccionar: la
pieza está colocada y sin movimiento, dando la mejor toma.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ParamsMotion:
    # historia y varThreshold de MOG2. history alto = fondo más estable ante
    # parpadeos de luz; varThreshold controla sensibilidad al cambio.
    history: int = 300
    var_threshold: float = 25.0
    detectar_sombras: bool = False
    # Fracción de píxeles de primer plano por encima de la cual consideramos
    # "hay movimiento / hay pieza entrando".
    umbral_movimiento: float = 0.02
    # Frames consecutivos por debajo del umbral para declarar "quieto" y
    # disparar la inspección una sola vez.
    frames_quieto: int = 12


class TriggerInspeccion:
    """Máquina de estados mínima: ESPERANDO_PIEZA -> PIEZA_MOVIENDO -> QUIETO.

    Se mantiene como objeto (no función pura) porque MOG2 y el conteo de frames
    quietos son estado que persiste entre cuadros; main.py conserva una
    instancia por sesión.
    """

    def __init__(self, params: ParamsMotion):
        self.params = params
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=params.history,
            varThreshold=params.var_threshold,
            detectShadows=params.detectar_sombras,
        )
        self.hubo_movimiento = False
        self.frames_estables = 0

    def procesar(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Devuelve (frame_anotado, {disparar: bool, ...}).

        disparar=True exactamente en el cuadro donde la pieza se acaba de
        quietar tras haberse movido; el llamador corre la inspección ahí.
        """
        fg = self.mog2.apply(frame)
        # Umbral binario: MOG2 marca sombras como gris (127) si estuvieran
        # activas; nos quedamos con primer plano fuerte (255).
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        frac = float(np.count_nonzero(fg_bin)) / fg_bin.size

        hay_mov = frac > self.params.umbral_movimiento
        disparar = False

        if hay_mov:
            self.hubo_movimiento = True
            self.frames_estables = 0
        else:
            # Sin movimiento: si antes SÍ lo hubo, contamos cuadros quietos
            # hasta confirmar que la pieza quedó colocada.
            if self.hubo_movimiento:
                self.frames_estables += 1
                if self.frames_estables >= self.params.frames_quieto:
                    disparar = True
                    self.hubo_movimiento = False  # rearmar para la próxima pieza
                    self.frames_estables = 0

        anotado = frame.copy()
        estado = "MOVIENDO" if hay_mov else ("QUIETO" if self.hubo_movimiento else "ESPERANDO")
        cv2.putText(
            anotado,
            f"{estado} fg={frac:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        if disparar:
            cv2.putText(
                anotado, "TRIGGER", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
            )
        return anotado, {"disparar": disparar, "frac_fg": frac, "estado": estado}
