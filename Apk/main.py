#!/usr/bin/env python3
"""
Pink Edge AI (Android) — Kivy UI. Single file, no .kv, matching this project's general "as few
files as possible" style. Mirrors GUI.py's workflow (pick/capture a scan -> run triage -> verdict +
bounding-box overlay -> save a text report) with mobile-appropriate widgets (Spinner instead of a
dropdown menu bar, plyer for the camera/gallery picker instead of tkinter.filedialog).

Model dispatch lives in mobile_inference.py — see that file's docstring and ../README.md's model
table for exactly which method backs which modality and why.
"""
import os
import time

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image as KivyImage
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.metrics import dp
from PIL import Image as PILImage, ImageDraw

import mobile_inference as inf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_PATH = os.path.join(APP_DIR, "_last_overlay.png")
REPORTS_DIR = os.path.join(APP_DIR, "reports")

MODALITY_LABELS = {
    "Mammography (Breast Cancer)": "mammography",
    "Tuberculosis (Chest X-Ray)": "tb",
    "Maternal Health (Ultrasound)": "maternal",
}

TR = {
    "en": {
        "title": "Pink Edge AI", "pick": "Pick / Capture Scan", "run": "Run Triage",
        "save": "Save Report", "settings": "Roboflow Key", "no_image": "No image selected yet.",
        "result_placeholder": "Run triage to see a verdict.", "lang": "EN",
    },
    "ur": {
        "title": "پنک ایج AI", "pick": "تصویر منتخب کریں", "run": "تشخیص کریں",
        "save": "رپورٹ محفوظ کریں", "settings": "روبوفلو کی", "no_image": "ابھی تک کوئی تصویر منتخب نہیں۔",
        "result_placeholder": "نتیجہ دیکھنے کے لیے تشخیص چلائیں۔", "lang": "UR",
    },
}


def draw_bbox_overlay(image_path, bbox, is_critical):
    """PIL-based equivalent of GUI.py's draw_bbox() -- draws the normalized (x, y, w, h) box (or a
    centered placeholder crosshair if bbox is None) and saves a copy Kivy can display."""
    img = PILImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    color = (220, 20, 60) if is_critical else (0, 150, 90)
    if bbox is not None:
        bx, by, bw, bh = bbox
        x0, y0 = bx * w, by * h
        x1, y1 = x0 + bw * w, y0 + bh * h
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, w // 200))
    else:
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.08
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=max(2, w // 200))
    img.save(OVERLAY_PATH)
    return OVERLAY_PATH


class PinkEdgeRoot(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=dp(12), spacing=dp(8), **kwargs)
        self.lang = "en"
        self.image_path = None
        self.last_result = None
        self.last_modality = None

        self.title_label = Label(text=TR[self.lang]["title"], font_size=dp(22), size_hint_y=None, height=dp(40))
        self.add_widget(self.title_label)

        self.modality_spinner = Spinner(
            text="Mammography (Breast Cancer)", values=list(MODALITY_LABELS.keys()),
            size_hint_y=None, height=dp(44))
        self.add_widget(self.modality_spinner)

        self.preview = KivyImage(size_hint_y=0.45)
        self.add_widget(self.preview)

        row1 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        self.pick_btn = Button(text=TR[self.lang]["pick"], on_press=self.on_pick)
        self.run_btn = Button(text=TR[self.lang]["run"], on_press=self.on_run, disabled=True)
        row1.add_widget(self.pick_btn)
        row1.add_widget(self.run_btn)
        self.add_widget(row1)

        self.result_label = Label(text=TR[self.lang]["result_placeholder"], size_hint_y=None,
                                   height=dp(90), halign="center", valign="middle")
        self.result_label.bind(size=lambda *_: setattr(self.result_label, "text_size", self.result_label.size))
        self.add_widget(self.result_label)

        row2 = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        self.save_btn = Button(text=TR[self.lang]["save"], on_press=self.on_save, disabled=True)
        self.lang_btn = Button(text=TR[self.lang]["lang"], size_hint_x=0.3, on_press=self.on_toggle_lang)
        self.settings_btn = Button(text="⚙", size_hint_x=0.2, on_press=self.on_settings)
        row2.add_widget(self.save_btn)
        row2.add_widget(self.lang_btn)
        row2.add_widget(self.settings_btn)
        self.add_widget(row2)

        self.status_label = Label(text="", size_hint_y=None, height=dp(24), font_size=dp(11))
        self.add_widget(self.status_label)

    # -- language ---------------------------------------------------------
    def on_toggle_lang(self, *_):
        self.lang = "ur" if self.lang == "en" else "en"
        t = TR[self.lang]
        self.title_label.text = t["title"]
        self.pick_btn.text = t["pick"]
        self.run_btn.text = t["run"]
        self.save_btn.text = t["save"]
        self.lang_btn.text = t["lang"]
        if not self.last_result:
            self.result_label.text = t["result_placeholder"]

    # -- image picking ------------------------------------------------------
    def on_pick(self, *_):
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=self._on_file_chosen, filters=["*.jpg", "*.jpeg", "*.png"])
        except Exception as exc:  # plyer backend unavailable (e.g. desktop test run without it)
            self.status_label.text = f"File picker unavailable ({exc}); use Settings to type a path."
            self._prompt_manual_path()

    def _prompt_manual_path(self):
        box = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        inp = TextInput(hint_text="/full/path/to/image.jpg", multiline=False)
        popup = Popup(title="Enter image path", content=box, size_hint=(0.9, 0.3))

        def _confirm(*_):
            self._on_file_chosen([inp.text.strip()])
            popup.dismiss()

        box.add_widget(inp)
        box.add_widget(Button(text="OK", size_hint_y=None, height=dp(40), on_press=_confirm))
        popup.open()

    def _on_file_chosen(self, selection):
        if not selection or not selection[0] or not os.path.isfile(selection[0]):
            return
        self.image_path = selection[0]
        self.preview.source = self.image_path
        self.preview.reload()
        self.run_btn.disabled = False
        self.last_result = None
        self.status_label.text = os.path.basename(self.image_path)

    # -- inference ------------------------------------------------------
    def on_run(self, *_):
        if not self.image_path:
            return
        modality = MODALITY_LABELS[self.modality_spinner.text]
        self.status_label.text = "Running triage..."
        try:
            pil_img = PILImage.open(self.image_path)
            result = inf.predict(modality, pil_img)
        except Exception as exc:
            self.result_label.text = f"Error: {exc}"
            self.status_label.text = ""
            return

        if result is None:
            self.result_label.text = ("No prediction available: no Roboflow key configured and no "
                                       "offline templates bundled for this modality.")
            self.status_label.text = ""
            return

        self.last_result = result
        self.last_modality = modality
        overlay = draw_bbox_overlay(self.image_path, result.get("bbox"), result["is_critical"])
        self.preview.source = overlay
        self.preview.reload()

        self.result_label.text = (
            f"{result['verdict']}  ({result['confidence']:.1f}%)\n{result['label']}\nSource: {result['source']}")
        self.save_btn.disabled = False
        self.status_label.text = ""

    # -- report -----------------------------------------------------------
    def on_save(self, *_):
        if not self.last_result:
            return
        os.makedirs(REPORTS_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(REPORTS_DIR, f"report-{self.last_modality}-{stamp}.txt")
        r = self.last_result
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"Pink Edge AI — Triage Report\nModality: {self.last_modality}\n"
                      f"Verdict: {r['verdict']}\nLabel: {r['label']}\n"
                      f"Confidence: {r['confidence']:.1f}%\nSMS payload: {r['sms']}\n"
                      f"Source: {r['source']}\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.status_label.text = f"Saved: {os.path.basename(path)}"

    # -- settings -----------------------------------------------------------
    def on_settings(self, *_):
        box = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))
        key_path = os.path.join(APP_DIR, "roboflow_key.txt")
        existing = ""
        if os.path.isfile(key_path):
            with open(key_path, "r", encoding="utf-8") as fh:
                existing = fh.read().strip()
        inp = TextInput(text=existing, hint_text="Roboflow API key", multiline=False, password=True)
        status_note = Label(text=inf.model_status(), font_size=dp(10), size_hint_y=None, height=dp(90))
        popup = Popup(title=TR[self.lang]["settings"], content=box, size_hint=(0.95, 0.6))

        def _save(*_):
            with open(key_path, "w", encoding="utf-8") as fh:
                fh.write(inp.text.strip())
            popup.dismiss()

        box.add_widget(inp)
        box.add_widget(Button(text="Save", size_hint_y=None, height=dp(40), on_press=_save))
        box.add_widget(ScrollView(size_hint_y=0.5).__class__() if False else status_note)  # status readout
        popup.open()


class PinkEdgeAndroidApp(App):
    def build(self):
        self.title = "Pink Edge AI"
        return PinkEdgeRoot()


if __name__ == "__main__":
    PinkEdgeAndroidApp().run()
