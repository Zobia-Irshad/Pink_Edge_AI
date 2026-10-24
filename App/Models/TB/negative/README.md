# Models/TB/negative/

Drop extra chest X-ray images here that you know are **healthy / no active TB** — no annotation file
needed. `offline_cv.py` picks these up automatically the next time it (re)builds its TB template,
alongside the labeled `Data Set/tuberculosis.coco` images, and auto-detects when you've added/removed
files here (no manual cache-clear needed).

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../positive/README.md` for the TB-positive
counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
