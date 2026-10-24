# Models/Mammography/validate/

Drop mammogram images here that you just want a quick prediction on — **no known label needed**,
this folder isn't used to build the template (see `../positive/` and `../negative/` for that). Run:

```
python offline_cv.py
```
(from `App/`) — it prints the offline heuristic's verdict for every image here, right after the
main calibration numbers, so you can eyeball how it handles new images without changing what the
app itself uses for accuracy reporting.

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`.
