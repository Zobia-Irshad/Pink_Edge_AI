# Models/TB/positive/

Drop extra chest X-ray images here that you know show **active tuberculosis** (or a TB-consistent
finding) — no annotation file needed. `offline_cv.py` picks these up automatically the next time it
(re)builds its TB template, alongside the labeled `Data Set/tuberculosis.coco` images, and
auto-detects when you've added/removed files here (no manual cache-clear needed).

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../negative/README.md` for the healthy
counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
