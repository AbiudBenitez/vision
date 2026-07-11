# Proyecto: Sistema de Inspección Visual de Fundas de Celular

Proyecto final de la materia **Visión Computacional** — FIME-UANL, grupo 002.
Docente: Delia Guadalupe Elizondo Sillas.

Aplicación en Python + OpenCV que inspecciona fundas de celular mediante webcam y
detecta defectos geométricos, aplicando técnicas clásicas de visión por computadora.

---

## Objetivo

Detectar defectos **geométricos/dimensionales** (NO defectos de superficie como
rayones o manchas) en fundas de celular:

- Barrenos de cámara faltantes, sobrantes o fuera de diámetro esperado
- Barrenos ovalados/deformados (deberían ser circulares)
- Esquinas dañadas, golpeadas o faltantes
- Bordes que no son rectos, o que perdieron paralelismo/ortogonalidad

**Razón del enfoque:** los defectos de superficie son un problema semántico que
requiere deep learning. Los defectos geométricos se resuelven con las técnicas
del temario y producen una demo confiable en vivo.

---

## Temas del temario que DEBEN quedar demostrados

Cada uno debe ser visible y explicable durante la exposición:

| Tema | Dónde se aplica en el proyecto |
|---|---|
| Transformada de Hough | Base común: `HoughLinesP` (bordes) y `HoughCircles` (barrenos) |
| Detección de círculos | Verificar presencia y diámetro de barrenos |
| Detección de elipses | `fitEllipse` sobre contornos → medir deformación de barrenos |
| Detección de agujeros | `findContours` con jerarquía (`RETR_CCOMP`) → contar barrenos |
| Polígonos y esquinas | `approxPolyDP` (contorno de la funda) + `goodFeaturesToTrack` / Harris |
| Detección de movimiento | `createBackgroundSubtractorMOG2` → trigger de inspección |

---

## Restricciones

- **OpenCV clásico únicamente. NO deep learning, NO redes neuronales.**
  Es requisito explícito del curso. Si una tarea parece necesitar una CNN,
  hay que replantear el problema, no cambiar de herramienta.
- Debe correr **en vivo con webcam**, no solo sobre imágenes guardadas.
- Condiciones de captura asumidas (controladas):
  - Iluminación fija, lateral (no luz de techo)
  - Fondo liso y contrastante respecto a la funda
  - Cámara perpendicular a la pieza y a distancia fija

---

## Estructura del proyecto

```
inspeccion-fundas/
├── CLAUDE.md
├── main.py              # loop de webcam + menú de modos
├── requirements.txt
├── src/
│   ├── preprocess.py    # grises, blur, threshold, morfología
│   ├── hough.py         # líneas y círculos
│   ├── shapes.py        # contornos, approxPolyDP, esquinas
│   ├── ellipses.py      # fitEllipse + criterio de deformación
│   ├── holes.py         # jerarquía de contornos
│   └── motion.py        # MOG2 / trigger de inspección
└── data/                # fotos de referencia (fundas buenas y defectuosas)
```

---

## Convenciones de código

- **Un módulo por técnica.** No mezclar detectores en un solo archivo.
- Funciones puras con firma consistente:
  ```python
  def detectar_x(frame, params) -> tuple[np.ndarray, dict]:
      """Devuelve (frame_anotado, resultados)."""
  ```
- Los parámetros de cada detector viven en un dict/dataclass, no hardcodeados
  dentro de la función. Facilita calibrar contra las fotos de `data/`.
- **Comentar el porqué del algoritmo**, no el qué. Los comentarios sirven de
  guion para la exposición.
  - Mal: `# aplica threshold`
  - Bien: `# Otsu elige el umbral automáticamente; sirve porque la iluminación
    es fija pero varía entre sesiones de captura`
- `main.py` no contiene lógica de visión: solo captura, despacha al módulo del
  modo activo, y dibuja.

---

## Orden de desarrollo sugerido

1. `preprocess.py` + `main.py` mostrando el threshold en vivo.
   **Calibrar iluminación aquí antes de escribir cualquier detector.**
2. `holes.py` — contornos y jerarquía (base para todo lo demás)
3. `hough.py` — círculos, luego líneas
4. `ellipses.py` — criterio de deformación (ratio de ejes)
5. `shapes.py` — polígonos y esquinas
6. `motion.py` — trigger de inspección
7. Integración: modo "inspección completa" que corre todo y emite veredicto

---

## Dataset

Guardar en `data/` entre 5 y 10 fotos:
- Fundas en buen estado (referencia)
- Fundas con defectos simulados: barreno tapado, esquina recortada,
  borde deformado

Sin ejemplos de defecto no se puede validar ningún detector.

---

## Notas para Claude

- Antes de agregar una dependencia nueva, preguntar. El stack debe ser
  mínimo: `opencv-python`, `numpy`.
- Al ajustar parámetros de un detector, hacerlo contra una foto concreta de
  `data/` y explicar qué rango se probó y por qué se eligió ese valor.
- Si un detector es frágil ante iluminación, decirlo explícitamente en vez de
  subir/bajar thresholds hasta que "funcione" con una sola imagen.
