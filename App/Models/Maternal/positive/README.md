# Models/Maternal/positive/

Drop extra fetal ultrasound images here that show an **abnormal finding** — no annotation file
needed. `offline_cv.py` picks these up automatically the next time it (re)builds its Maternal
template, alongside the labeled `Data Set/HASH Maternal Health.coco` images, and auto-detects when
you've added/removed files here (no manual cache-clear needed).

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../negative/README.md` for the
normal-finding counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
