# data/ — dataset de calibración

Faltan las fotos. Sin ejemplos de defecto **no se puede validar ningún detector**
(ver CLAUDE.md). Guardar aquí entre 5 y 10 imágenes:

- **Fundas buenas** (referencia): `buena_01.jpg`, `buena_02.jpg`, ...
- **Con defectos simulados**:
  - barreno tapado → valida `holes.py` (conteo)
  - barreno ovalado → valida `ellipses.py` (ratio de ejes)
  - esquina recortada → valida `shapes.py` (conteo de vértices)
  - borde deformado → valida `hough.py` (líneas)

## Cómo capturar

Condiciones controladas (CLAUDE.md):
- iluminación fija lateral (no luz de techo)
- fondo liso y contrastante contra la funda
- cámara perpendicular, distancia fija

Capturar rápido desde la app: `python main.py`, tecla `1` (preprocess), ajustar
luz hasta que la silueta binaria quede limpia y cerrada. Recién ahí tomar las
fotos y probar cada detector con `python main.py --img data/<foto>.jpg`.
