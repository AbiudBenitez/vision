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
    # Los umbrales de área son FRACCIONES, no píxeles. Una foto de celular
    # (4000x3000) y la webcam de la demo (640x480) difieren ~40x en área, así
    # que cualquier umbral en px^2 calibrado contra data/ sería inservible en
    # vivo. Las fracciones son invariantes a la resolución: se calibran una vez
    # con las fotos y siguen valiendo frente a la webcam.

    # Fracción mínima del FRAME para aceptar un contorno exterior como la funda.
    # Medido en data/: la funda ocupa 23%-25% del frame. 0.02 deja margen amplio
    # para que la pieza pueda estar más lejos sin dejar de detectarse.
    frac_min_funda: float = 0.02
    # Fracción del ÁREA DE REFERENCIA de la funda (ver area_referencia) que debe
    # medir un hueco para contar como barreno.
    #
    # Medido contra data/ Y contra la webcam, usando el rectángulo envolvente como
    # denominador: los barrenos sanos caen en 0.0046-0.0059 en AMBOS montajes. El
    # barreno defectuoso de def_ovalado mide 0.0023 y el ruido del estampado que
    # sobrevive al filtro de circularidad se queda en 0.0010-0.0014.
    #
    # El piso 0.0018 se planta entre el barreno defectuoso (0.0023) y el ruido
    # (0.0014). NO se sube hasta pegarlo al rango sano a propósito: si el barreno
    # deforme se cae de este filtro, el sistema lo reporta como barreno FALTANTE
    # en vez de DEFORME -> diagnóstico equivocado, y ellipses.py nunca lo ve.
    # El techo 0.008 deja ~1.35x de holgura sobre el barreno sano más grande.
    frac_barreno_min: float = 0.0018
    frac_barreno_max: float = 0.008
    # Circularidad 4*pi*A/P^2 (1.0 = círculo perfecto). Este es el filtro que
    # hace el trabajo pesado, no el área: al bajar frac_barreno_min a 0.002
    # entran manchas del estampado en la banda 0.002-0.003, pero TODAS miden
    # 0.26-0.45 de circularidad, mientras que los barrenos (sanos o deformes)
    # miden 0.82-0.91. El corte en 0.60 cae en medio de ese abismo: ninguna
    # mancha se le acerca y ningún barreno se le escapa.
    circularidad_min: float = 0.60
    # Barrenos esperados en una funda buena: los 3 de la cámara. El veredicto
    # compara el conteo real contra esto. En data/, def_tapado da 2 -> NG.
    barrenos_esperados: int = 3


def _circularidad(contorno) -> float:
    """4*pi*A/P^2: 1.0 para un círculo, tiende a 0 para formas alargadas o de
    perímetro rugoso. Distingue un barreno de una mancha del estampado que
    casualmente tenga el área correcta."""
    perimetro = cv2.arcLength(contorno, True)
    if perimetro <= 0:
        return 0.0
    return float(4.0 * np.pi * cv2.contourArea(contorno) / (perimetro * perimetro))


def area_referencia(contorno) -> float:
    """Tamaño de la funda, medido con su rectángulo envolvente (minAreaRect).

    Es el DENOMINADOR con el que se normaliza todo lo demás (tamaño de barreno,
    radio esperado), así que tiene que ser estable. El área del contorno NO lo es:
    la funda va estampada, y cuando el dibujo claro alcanza el borde de la pieza,
    la binarización le muerde la silueta. Medido en webcam, esa mordida hunde el
    área del contorno a 16% del frame cuando la pieza realmente ocupa 31%.

    Eso envenena todo lo que se normalice con ella: frac_barreno = área_barreno /
    área_funda, así que un denominador hundido INFLA la fracción y expulsa a
    barrenos perfectamente sanos (medido: 0.0113 contra un techo de 0.010).

    El rectángulo envolvente no se hunde. La funda es un rectángulo, y su
    rectángulo envolvente mide lo mismo aunque al contorno le falte un pedazo.
    Con este denominador los barrenos sanos dan 0.0046-0.0059 tanto en las fotos
    como en la webcam con reflejo; con el área del contorno, 0.0050 y 0.0113.
    """
    (_, (w, h), _) = cv2.minAreaRect(contorno)
    return float(w * h)


def rectangularidad(contorno) -> float:
    """Área del contorno / área de su rectángulo envolvente. 1.0 = rectángulo
    perfecto.

    Mide cuánto se aleja la silueta de un rectángulo, y sirve como SEMÁFORO de
    confianza: una funda bien segmentada da 0.90-0.96, pero si el estampado o un
    reflejo le muerden la silueta se desploma (medido: 0.52 en webcam). Cuando
    cae, el conteo de vértices y de esquinas deja de ser fiable, y más vale
    avisarlo que reportar un veredicto de forma en el que no se puede confiar.
    """
    a_rect = area_referencia(contorno)
    return float(cv2.contourArea(contorno) / a_rect) if a_rect > 0 else 0.0


def indices_barrenos(contornos, jerarquia, i_funda, params: "ParamsHoles") -> list[int]:
    """Índices de los hijos de la funda que son barrenos de verdad.

    Los barrenos son los hijos directos del contorno de la funda: recorremos la
    cadena de hermanos que cuelgan de i_funda (jerarquia[i][2] = primer hijo,
    jerarquia[i][0] = siguiente hermano).

    Ojo: en una funda ESTAMPADA los hijos no son solo los barrenos. El dibujo
    impreso tiene zonas oscuras que la binarización convierte en huecos, y
    aparecen aquí como hermanos legítimos (~30-40 por foto en data/). Por eso el
    conteo crudo de hijos no sirve: hay que filtrarlos por tamaño y forma.

    Vive aquí y es pública porque ellipses.py debe juzgar EXACTAMENTE los mismos
    huecos que holes.py cuenta. Si cada módulo eligiera sus barrenos por su
    cuenta, el veredicto podría decir "3 barrenos OK" mientras el de elipses mide
    la deformación de una mancha del estampado.
    """
    a_ref = area_referencia(contornos[i_funda])
    if a_ref <= 0:
        return []

    barrenos = []
    hijo = jerarquia[0][i_funda][2]
    while hijo != -1:
        c = contornos[hijo]
        frac = cv2.contourArea(c) / a_ref
        if (
            params.frac_barreno_min <= frac <= params.frac_barreno_max
            and _circularidad(c) >= params.circularidad_min
        ):
            barrenos.append(hijo)
        hijo = jerarquia[0][hijo][0]
    return barrenos


def _toca_borde(contorno, forma, margen: int = 2) -> bool:
    """True si el contorno llega al borde del frame.

    Sirve para reconocer al FONDO y descartarlo. La funda se inspecciona
    completa, así que por construcción está entera dentro del encuadre y no
    puede tocar el borde; cualquier cosa que sí lo toque está recortada por el
    marco y no es la pieza.
    """
    alto, ancho = forma[:2]
    x, y, w, h = cv2.boundingRect(contorno)
    return (
        x <= margen
        or y <= margen
        or x + w >= ancho - margen
        or y + h >= alto - margen
    )


def _contorno_funda(contornos, jerarquia, area_min, forma=None):
    """Regresa el índice del contorno exterior que corresponde a la funda.

    En RETR_CCOMP los exteriores tienen jerarquia[i][3] == -1 (sin padre).

    "El exterior más grande" NO basta como criterio, y falla en cuanto el fondo
    deja de ser perfecto. Medido con la webcam: al colarse un teclado y un
    monitor oscuros en el encuadre, la binarización los une en un blob que ocupa
    el frame entero (bbox 0,0,640x360) y le gana por área a la funda (8.6%). El
    detector se quedaba mirando ese blob, que no tiene barrenos, y reportaba
    "0 barrenos" sobre una funda perfectamente sana.

    Por eso se descarta primero lo que toca el borde del frame: el fondo siempre
    lo toca y la pieza, que se inspecciona completa, nunca. Entre lo que queda,
    el más grande sí es la funda.

    `forma` es (alto, ancho) del frame. Si no se pasa, se conserva el criterio
    viejo de solo-área, para no romper llamadores que aún no lo proveen.
    """
    mejor_i, mejor_area = -1, 0.0
    for i, c in enumerate(contornos):
        if jerarquia[0][i][3] != -1:  # tiene padre => es un hueco, no la funda
            continue
        if forma is not None and _toca_borde(c, forma):
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

    area_frame = float(binaria.shape[0] * binaria.shape[1])
    i_funda = _contorno_funda(
        contornos, jerarquia, params.frac_min_funda * area_frame, binaria.shape
    )
    if i_funda == -1:
        return anotado, {
            "barrenos": 0, "ok": False, "motivo": "sin_funda", "area_funda": 0.0,
        }

    cv2.drawContours(anotado, contornos, i_funda, (255, 0, 0), 2)

    barrenos = indices_barrenos(contornos, jerarquia, i_funda, params)
    for i in barrenos:
        cv2.drawContours(anotado, contornos, i, (0, 0, 255), 2)

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
        # Hough lo necesita para saber qué radio de barreno esperar; se publica
        # aquí para no volver a segmentar la funda por segunda vez. Es el área de
        # REFERENCIA (rectángulo envolvente), no la del contorno: ver
        # area_referencia() para por qué la del contorno no sirve como escala.
        "area_funda": area_referencia(contornos[i_funda]),
        # Semáforo de confianza en la silueta: si baja, la forma no es fiable.
        "rectangularidad": rectangularidad(contornos[i_funda]),
    }
