"""Configuración central de la inspección: una sola fuente de verdad.

Antes cada modo y la inspección completa creaban sus propios params por
defecto, así que calibrar un modo NO afectaba al veredicto (divergían). Aquí se
agrupan todas las sub-params en un objeto; main.py crea UNA instancia y la
comparte. Calibrar una vez se propaga a todos.
"""
from dataclasses import dataclass, field

from .preprocess import ParamsPreproceso
from .holes import ParamsHoles
from .hough import ParamsCirculos, ParamsLineas
from .ellipses import ParamsElipses
from .shapes import ParamsShapes
from .motion import ParamsMotion


@dataclass
class ParamsInspeccion:
    # Ancho de trabajo: TODO frame se normaliza a esta anchura antes de tocar
    # cualquier detector. Los parámetros del proyecto se calibraron contra las
    # fotos de data/ reducidas a 640px; procesar a otra escala cambia cuánto
    # detalle del estampado sobrevive a la binarización y descalibra los
    # detectores. Cambiar este número obliga a recalibrar todo.
    ancho_trabajo: int = 640

    pre: ParamsPreproceso = field(default_factory=ParamsPreproceso)
    holes: ParamsHoles = field(default_factory=ParamsHoles)
    circulos: ParamsCirculos = field(default_factory=ParamsCirculos)
    lineas: ParamsLineas = field(default_factory=ParamsLineas)
    ellipses: ParamsElipses = field(default_factory=ParamsElipses)
    shapes: ParamsShapes = field(default_factory=ParamsShapes)
    motion: ParamsMotion = field(default_factory=ParamsMotion)

    def __post_init__(self):
        # Los detectores que binarizan comparten la MISMA config de
        # preprocesamiento: así calibrar la iluminación una sola vez llega a
        # holes, ellipses y shapes por igual, y el veredicto usa exactamente lo
        # que se ve en cada modo individual.
        self.holes.pre = self.pre
        self.ellipses.pre = self.pre
        self.shapes.pre = self.pre
        # ellipses debe juzgar EXACTAMENTE los barrenos que holes cuenta, así
        # que comparte su criterio de selección en vez de tener uno propio.
        self.ellipses.holes = self.holes
