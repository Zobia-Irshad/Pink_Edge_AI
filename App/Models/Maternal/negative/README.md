# Models/Maternal/negative/

Drop extra fetal ultrasound images here that show a **normal finding** — no annotation file needed.
`offline_cv.py` picks these up automatically the next time it (re)builds its Maternal template,
alongside the labeled `Data Set/HASH Maternal Health.coco` images, and auto-detects when you've
added/removed files here (no manual cache-clear needed). This is especially useful for Maternal
specifically — the source dataset is single-class (only "abnormal" is annotated; unannotated images
are used as an implicit negative set), so real, deliberately-labeled normal images added here
directly improve that side of the template.

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../positive/README.md` for the
abnormal-finding counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
