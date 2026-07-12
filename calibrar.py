"""Herramienta de calibración: mide fotos de data/ y sugiere valores de config.

Problema que resuelve: los parámetros de src/config.py estaban adivinados. En
vez de mover thresholds a ciegas hasta que "funcione" con una foto (lo que
CLAUDE.md prohíbe), este script MIDE las fotos reales y vuelca números
concretos —radios de barrenos, ratios de ejes, vértices, fracciones de área—
para llenar config.py con evidencia.

Dos reglas que este script respeta y conviene no romper:

1. Mide a `ancho_trabajo` (640px), la MISMA escala a la que corren los
   detectores en vivo. Medir a la resolución nativa del celular (4000px) daría
   números que no valen para la webcam: a esa escala la binarización saca ~86
   huecos del estampado contra ~34 a 640px.

2. Cuenta barrenos con `holes.indices_barrenos`, el criterio REAL del detector,
   no con un umbral propio. Antes usaba un `area >= 50` suyo y reportaba 36-93
   "barrenos" (estaba contando el estampado), lo que hacía que sugiriera
   `barrenos_esperados = 86`.

No modifica nada: solo imprime un reporte por foto y, al final, un bloque de
valores sugeridos. Las fotos cuyo nombre empieza con "buena" fijan los rangos
esperados; las demás (defectos) solo se reportan para comprobar que caen fuera.

Uso:
  .venv/bin/python calibrar.py                 # todas las fotos de data/
  .venv/bin/python calibrar.py data/una.jpg    # una foto puntual
"""
import glob
import statistics
import sys

import cv2
import numpy as np

from src.config import ParamsInspeccion
from src.preprocess import ParamsPreproceso, binarizar, escalar_a_ancho
from src.holes import _circularidad, _contorno_funda, indices_barrenos

# Una "funda" que ocupa casi todo el frame no es la funda: es el fondo mal
# segmentado por la polaridad equivocada. Medido en data/: la funda real ocupa
# 22-25% del frame y el fondo, con la polaridad invertida, ocupa 94-97%. El corte
# en 80% separa ambos casos con muchísimo margen.
FRAC_MAX_FUNDA = 0.80


def _distancia_min_centros(centros):
    """Menor distancia entre centros de barrenos: acota HoughCircles.min_dist
    para que no fusione dos barrenos cercanos en un solo círculo."""
    if len(centros) < 2:
        return None
    return min(
        float(np.hypot(centros[i][0] - centros[j][0], centros[i][1] - centros[j][1]))
        for i in range(len(centros))
        for j in range(i + 1, len(centros))
    )


def medir_una_polaridad(frame, invertir, cfg):
    """Mide la foto asumiendo una polaridad de binarización.

    Devuelve None si no aparece una funda plausible, lo que es la señal de que
    esta polaridad es la equivocada.
    """
    binaria, _ = binarizar(frame, ParamsPreproceso(invertir=invertir))
    total = float(binaria.shape[0] * binaria.shape[1])

    contornos, jer = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if jer is None or len(contornos) == 0:
        return None

    i_funda = _contorno_funda(contornos, jer, cfg.holes.frac_min_funda * total)
    if i_funda == -1:
        return None

    area_funda = cv2.contourArea(contornos[i_funda])
    if area_funda > FRAC_MAX_FUNDA * total:
        return None  # es el fondo, no la funda

    # Barrenos según el criterio REAL del detector, no uno propio.
    barrenos = []
    for i in indices_barrenos(contornos, jer, i_funda, cfg.holes):
        c = contornos[i]
        if len(c) < 5:
            continue
        (cx, cy), (e1, e2), _ = cv2.fitEllipse(c)
        menor, mayor = sorted((e1, e2))
        barrenos.append({
            "frac": cv2.contourArea(c) / area_funda,
            "circ": _circularidad(c),
            "ratio": menor / mayor if mayor > 0 else 0.0,
            "radio": 0.5 * mayor,  # radio aparente ~ semieje mayor
            "centro": (cx, cy),
        })

    eps = cfg.shapes.epsilon_frac * cv2.arcLength(contornos[i_funda], True)
    return {
        "invertir": invertir,
        "frac_funda": area_funda / total,
        "n_barrenos": len(barrenos),
        "barrenos": barrenos,
        "vertices": len(cv2.approxPolyDP(contornos[i_funda], eps, True)),
    }


def medir(frame, cfg):
    """Prueba ambas polaridades y devuelve la plausible.

    Ojo con el criterio: NO se elige "la de mayor área". Esa heurística elegía
    sistemáticamente la polaridad equivocada, porque el fondo mal segmentado
    ocupa 94-97% del frame y siempre gana por área a la funda real (22-25%).
    El filtro FRAC_MAX_FUNDA descarta al fondo antes de comparar; entre lo que
    sobrevive, más barrenos = mejor segmentación.
    """
    candidatos = [
        m for m in (medir_una_polaridad(frame, inv, cfg) for inv in (True, False)) if m
    ]
    if not candidatos:
        return None
    return max(candidatos, key=lambda m: (m["n_barrenos"], m["frac_funda"]))


def main():
    rutas = sys.argv[1:] or sorted(
        glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")
    )
    if not rutas:
        print("No hay fotos en data/. Agrega buena_01.jpg, def_*.jpg, etc.")
        return 1

    cfg = ParamsInspeccion()
    mediciones = []
    for ruta in rutas:
        frame = cv2.imread(ruta)
        if frame is None:
            print(f"[SKIP] no se pudo leer {ruta}")
            continue
        frame = escalar_a_ancho(frame, cfg.ancho_trabajo)

        m = medir(frame, cfg)
        print(f"\n=== {ruta}  ({frame.shape[1]}x{frame.shape[0]}) ===")
        if m is None:
            print("  no se detectó funda (revisa contraste fondo/funda e iluminación)")
            continue

        pol = "funda oscura (invertir=True)" if m["invertir"] else "funda clara (invertir=False)"
        print(f"  polaridad : {pol}")
        print(f"  funda     : {m['frac_funda']:.1%} del frame")
        print(f"  vertices  : {m['vertices']}")
        print(f"  barrenos  : {m['n_barrenos']}")
        for i, b in enumerate(m["barrenos"]):
            print(
                f"    #{i}: radio={b['radio']:.1f}px  ratio_ejes={b['ratio']:.2f}"
                f"  frac_area={b['frac']:.4f}  circ={b['circ']:.2f}"
            )
        dmin = _distancia_min_centros([b["centro"] for b in m["barrenos"]])
        if dmin is not None:
            print(f"  dist min entre centros: {dmin:.0f}px")
        mediciones.append((ruta, m))

    # Sugerencias: solo desde fotos "buena_*", que definen lo NORMAL. Los
    # defectos deben quedar FUERA de estos rangos, no ampliarlos.
    buenas = [m for r, m in mediciones if r.split("/")[-1].lower().startswith("buena")]
    if not buenas:
        print(
            "\n[!] Sin fotos 'buena_*' no se pueden sugerir rangos. "
            "Nombra al menos una funda sana como buena_01.jpg y reejecuta."
        )
        return 0

    if len({m["invertir"] for m in buenas}) > 1:
        print(
            "\n[!] Las fotos buenas dieron polaridades distintas: encuadre o luz "
            "inconsistentes. Rehaz las fotos con el mismo montaje."
        )

    ancho = cfg.ancho_trabajo
    radios = [b["radio"] for m in buenas for b in m["barrenos"]]
    ratios = [b["ratio"] for m in buenas for b in m["barrenos"]]
    fracs = [b["frac"] for m in buenas for b in m["barrenos"]]
    dmins = [
        d for m in buenas
        if (d := _distancia_min_centros([b["centro"] for b in m["barrenos"]])) is not None
    ]

    print("\n" + "=" * 62)
    print(f"VALORES SUGERIDOS para src/config.py (medidos a {ancho}px de ancho)")
    print("=" * 62)
    print(f"# pre.invertir              = {buenas[0]['invertir']}")
    print(f"# holes.barrenos_esperados  = {statistics.mode(m['n_barrenos'] for m in buenas)}")
    print(f"# shapes.vertices_esperados = {statistics.mode(m['vertices'] for m in buenas)}")
    print(f"# holes.frac_min_funda      = {0.5 * min(m['frac_funda'] for m in buenas):.3f}")
    if fracs:
        print(f"# holes.frac_barreno_max    = {1.7 * max(fracs):.4f}")
    if radios:
        print(f"# circulos.frac_radio_min   = {0.8 * min(radios) / ancho:.4f}"
              f"   (radio sano min={min(radios):.1f}px)")
        print(f"# circulos.frac_radio_max   = {1.25 * max(radios) / ancho:.4f}"
              f"   (radio sano max={max(radios):.1f}px)")
    if dmins:
        print(f"# circulos.frac_min_dist    = {0.75 * min(dmins) / ancho:.4f}")
    if ratios:
        print(f"# ellipses.ratio_min        = {min(ratios) - 0.02:.2f}"
              f"   (peor ratio sano={min(ratios):.2f})")
    print("=" * 62)
    print("OJO: frac_barreno_min NO se sugiere a partir de las fotos buenas.")
    print("Debe quedar por DEBAJO del barreno defectuoso más chico que quieras")
    print("diagnosticar como 'deforme'; si lo pegas al rango sano, un barreno")
    print("encogido se cae del filtro y el sistema lo reporta como FALTANTE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
