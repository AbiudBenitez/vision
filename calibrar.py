"""Herramienta de calibración: mide fotos de data/ y sugiere valores de config.

Problema que resuelve: los parámetros de src/config.py estaban adivinados. En
vez de mover thresholds a ciegas hasta que "funcione" con una foto (lo que
CLAUDE.md prohíbe), este script MIDE las fotos reales y vuelca números
concretos —radios de barrenos, ratios de ejes, vértices, áreas— para llenar
config.py con evidencia.

No modifica nada: solo imprime un reporte por foto y, al final, un bloque de
valores sugeridos listo para pegar. Las fotos cuyo nombre empieza con "buena"
se usan para fijar los rangos esperados; las demás (defectos) solo se reportan
para comprobar que salen fuera de rango.

Uso:
  .venv/bin/python calibrar.py                 # todas las fotos de data/
  .venv/bin/python calibrar.py data/una.jpg    # una foto puntual
"""
import glob
import statistics
import sys

import cv2
import numpy as np

from src.preprocess import ParamsPreproceso, binarizar
from src.holes import _contorno_funda


def _distancia_min_centros(centros):
    """Menor distancia entre centros de barrenos: acota HoughCircles.min_dist
    para que no fusione dos barrenos cercanos en un solo círculo."""
    if len(centros) < 2:
        return None
    d = min(
        float(np.hypot(centros[i][0] - centros[j][0], centros[i][1] - centros[j][1]))
        for i in range(len(centros))
        for j in range(i + 1, len(centros))
    )
    return d


def medir_una_polaridad(frame, invertir):
    """Mide la foto asumiendo una polaridad de binarización. Devuelve None si
    no aparece una funda válida (contorno grande que no sea el borde del frame),
    lo que sirve para descartar la polaridad equivocada."""
    pre = ParamsPreproceso(invertir=invertir)
    binaria, _ = binarizar(frame, pre)
    total = binaria.shape[0] * binaria.shape[1]

    contornos, jer = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if jer is None:
        return None
    i_funda = _contorno_funda(contornos, jer, area_min=0.02 * total)
    if i_funda == -1:
        return None

    area_funda = cv2.contourArea(contornos[i_funda])
    # Si la "funda" ocupa casi todo el frame, es fondo mal segmentado: descartar.
    if area_funda > 0.95 * total:
        return None

    # Barrenos = hijos del contorno de la funda; mide área y ratio de ejes.
    barrenos = []
    hijo = jer[0][i_funda][2]
    while hijo != -1:
        c = contornos[hijo]
        area = cv2.contourArea(c)
        if area >= 50 and len(c) >= 5:
            (cx, cy), (e1, e2), _ = cv2.fitEllipse(c)
            menor, mayor = sorted((e1, e2))
            ratio = menor / mayor if mayor > 0 else 0.0
            radio = 0.5 * mayor  # radio aparente ~ semieje mayor
            barrenos.append(
                {"area": area, "ratio": ratio, "radio": radio, "centro": (cx, cy)}
            )
        hijo = jer[0][hijo][0]

    # Vértices del contorno de la funda con el mismo epsilon que shapes.py.
    eps = 0.02 * cv2.arcLength(contornos[i_funda], True)
    vertices = len(cv2.approxPolyDP(contornos[i_funda], eps, True))

    return {
        "invertir": invertir,
        "area_funda": area_funda,
        "n_barrenos": len(barrenos),
        "barrenos": barrenos,
        "vertices": vertices,
    }


def medir(frame):
    """Prueba ambas polaridades y devuelve la que dé la funda más plausible
    (mayor área válida). Así el script deduce solo si la funda es clara u
    oscura, sin que el usuario lo configure de antemano."""
    candidatos = [m for m in (medir_una_polaridad(frame, inv) for inv in (True, False)) if m]
    if not candidatos:
        return None
    return max(candidatos, key=lambda m: m["area_funda"])


def main():
    rutas = sys.argv[1:] or sorted(
        glob.glob("data/*.jpg") + glob.glob("data/*.jpeg") + glob.glob("data/*.png")
    )
    if not rutas:
        print("No hay fotos en data/. Agrega buena_01.jpg, def_*.jpg, etc.")
        return 1

    mediciones = []  # (ruta, medicion) de fotos legibles
    for ruta in rutas:
        frame = cv2.imread(ruta)
        if frame is None:
            print(f"[SKIP] no se pudo leer {ruta}")
            continue
        m = medir(frame)
        print(f"\n=== {ruta} ===")
        if m is None:
            print("  no se detectó funda (revisa contraste fondo/funda e iluminación)")
            continue

        pol = "funda oscura (invertir=True)" if m["invertir"] else "funda clara (invertir=False)"
        print(f"  polaridad : {pol}")
        print(f"  area funda: {m['area_funda']:.0f} px^2")
        print(f"  vertices  : {m['vertices']}")
        print(f"  barrenos  : {m['n_barrenos']}")
        for i, b in enumerate(m["barrenos"]):
            print(
                f"    #{i}: radio={b['radio']:.1f}px  ratio_ejes={b['ratio']:.2f}"
                f"  area={b['area']:.0f}"
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

    inversiones = {m["invertir"] for m in buenas}
    if len(inversiones) > 1:
        print(
            "\n[!] Las fotos buenas dieron polaridades distintas: encuadre o luz "
            "inconsistentes. Rehaz las fotos con el mismo montaje."
        )

    radios = [b["radio"] for m in buenas for b in m["barrenos"]]
    ratios = [b["ratio"] for m in buenas for b in m["barrenos"]]
    areas_b = [b["area"] for m in buenas for b in m["barrenos"]]
    n_barr = [m["n_barrenos"] for m in buenas]
    verts = [m["vertices"] for m in buenas]
    areas_funda = [m["area_funda"] for m in buenas]
    dmins = [d for m in buenas if (d := _distancia_min_centros([b["centro"] for b in m["barrenos"]])) is not None]

    print("\n" + "=" * 60)
    print("VALORES SUGERIDOS para src/config.py (revisa antes de pegar):")
    print("=" * 60)
    print(f"# pre.invertir      = {list(inversiones)[0]}")
    print(f"# holes.barrenos_esperados = {statistics.mode(n_barr)}")
    print(f"# shapes.vertices_esperados = {statistics.mode(verts)}")
    print(f"# holes.area_min_funda     = {0.5 * min(areas_funda):.0f}")
    if areas_b:
        print(f"# holes.area_min_barreno   = {0.5 * min(areas_b):.0f}")
    if radios:
        print(f"# circulos.radio_min = {max(1, int(0.7 * min(radios)))}")
        print(f"# circulos.radio_max = {int(1.3 * max(radios))}")
    if dmins:
        print(f"# circulos.min_dist  = {0.8 * min(dmins):.0f}")
    if ratios:
        # ratio_min por debajo del peor barreno bueno: un óvalo defectuoso
        # caerá aún más abajo y se marcará.
        print(f"# ellipses.ratio_min = {max(0.5, min(ratios) - 0.05):.2f}"
              f"   (peor ratio bueno={min(ratios):.2f})")
    print("=" * 60)
    print("Margen 0.7x/1.3x en radios y 0.5x en áreas: tolera variación entre")
    print("piezas sin dejar pasar defectos. Ajusta si un defecto no se marca.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
