# Sistema de Inspección Visual de Fundas de Celular

Proyecto final de **Visión Computacional** — FIME-UANL, grupo 002.
Docente: Delia Guadalupe Elizondo Sillas.

Aplicación en **Python + OpenCV clásico** que inspecciona fundas de celular por
webcam y detecta defectos **geométricos/dimensionales** (barrenos, esquinas,
bordes) — **sin deep learning**, por requisito del curso.

> Los defectos de superficie (rayones, manchas) son un problema semántico que
> exige redes neuronales. Los geométricos se resuelven con las técnicas del
> temario y dan una demo confiable en vivo.

---

## Defectos que detecta

- Barrenos de cámara faltantes, sobrantes o fuera de diámetro
- Barrenos ovalados/deformados (deberían ser circulares)
- Esquinas dañadas o faltantes
- Bordes no rectos o sin ortogonalidad

## Temas del temario → dónde se demuestran

| Tema | Módulo | Función clave |
|---|---|---|
| Transformada de Hough | `src/hough.py` | `HoughCircles`, `HoughLinesP` |
| Detección de círculos | `src/hough.py` | `HoughCircles` (presencia + diámetro) |
| Detección de elipses | `src/ellipses.py` | `fitEllipse` (ratio de ejes) |
| Detección de agujeros | `src/holes.py` | `findContours` + `RETR_CCOMP` |
| Polígonos y esquinas | `src/shapes.py` | `approxPolyDP`, `goodFeaturesToTrack` |
| Detección de movimiento | `src/motion.py` | `createBackgroundSubtractorMOG2` |

---

## Instalación

Requiere Python 3.10+. Stack mínimo: `opencv-python`, `numpy`.

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py                  # webcam (cámara 0)
python main.py --cam 1          # otra webcam
python main.py --img data/x.jpg # una foto fija (para calibrar/validar)
```

### Teclas

| Tecla | Modo | | Tecla | Modo |
|---|---|---|---|---|
| `1` | preprocess (calibrar luz) | | `6` | shapes (polígono/esquinas) |
| `2` | holes (contar barrenos) | | `7` | motion (trigger) |
| `3` | circulos (Hough) | | `8` | inspección completa |
| `4` | lineas (Hough) | | `i` | inspeccionar frame actual |
| `5` | ellipses (deformación) | | `q` | salir |

El modo **inspección completa** combina holes + shapes + ellipses + círculos +
líneas y emite un veredicto `ACEPTADA` / `RECHAZADA` con 5 checks.

---

## Estructura

```
inspeccion-fundas/
├── main.py            # loop de webcam + menú de modos (sin lógica de visión)
├── requirements.txt
├── src/
│   ├── config.py      # ParamsInspeccion: fuente única de parámetros
│   ├── preprocess.py  # grises, blur, Otsu, morfología
│   ├── holes.py       # jerarquía de contornos (RETR_CCOMP)
│   ├── hough.py       # HoughCircles + HoughLinesP
│   ├── ellipses.py    # fitEllipse + criterio de deformación
│   ├── shapes.py      # approxPolyDP + esquinas Shi-Tomasi
│   ├── motion.py      # MOG2 + trigger de inspección
│   └── inspeccion.py  # modo integración + veredicto
└── data/              # fotos de referencia (buenas y defectuosas)
```

Convención: un módulo por técnica; todos los detectores comparten la firma
`detectar_x(frame, params, binaria=None) -> (frame_anotado, resultados)`.

---

## Calibración (IMPORTANTE)

Los parámetros por defecto en `src/config.py` **no están calibrados**: son
puntos de partida razonables. El flujo correcto:

1. Condiciones controladas: iluminación fija **lateral** (no de techo), fondo
   liso y contrastante, cámara perpendicular a distancia fija.
2. `python main.py`, tecla `1`: ajustar luz hasta que la silueta binaria quede
   **limpia y cerrada**. Sin esto ningún detector es confiable.
3. Tomar 5–10 fotos a `data/` (buenas + defectos simulados: barreno tapado,
   barreno ovalado, esquina recortada, borde deformado). Ver `data/README.md`.
4. Calibrar cada detector con `python main.py --img data/<foto>.jpg`, ajustando
   sus rangos en `src/config.py` (`radio_min/max`, `ratio_min`,
   `barrenos_esperados`, `vertices_esperados`, `area_min_*`).

## Estado / pendientes

- [ ] Fotos de dataset en `data/` (bloqueante: sin defectos no se valida nada).
- [ ] Calibrar parámetros contra fotos reales.
- [ ] `bordes_ok` (ortogonalidad en `inspeccion.py`) es un criterio grueso;
      endurecer con paralelismo real tras calibrar.

---

## Restricciones del curso

OpenCV clásico únicamente. **NO deep learning, NO redes neuronales.** Debe
correr en vivo con webcam, no solo sobre imágenes guardadas.
