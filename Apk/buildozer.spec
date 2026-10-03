[app]
title = Pink Edge AI
package.name = pinkedgeai
package.domain = org.pinkedgeai

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,txt

version = 0.1.0

# kivy: UI toolkit. requests: Roboflow REST calls. pillow/numpy: offline pixel-diff heuristic
# (no opencv, no torch -- neither has a dependable python-for-android recipe / reasonable size).
# plyer: camera + filesystem access shims across Android versions.
requirements = python3,kivy==2.3.0,requests,pillow,numpy,plyer

# Portrait matches how a phone/tablet is actually held for this kind of form-fill UI.
orientation = portrait
fullscreen = 0

icon.filename = assets/icon.png
presplash.filename = assets/presplash.png

android.permissions = INTERNET,CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# API 31+ is required by current Play/AOSP toolchains; minapi 24 keeps it usable on older rural
# clinic devices (Android 7.0+, released 2016) without dropping current-toolchain compatibility.
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a,armeabi-v7a

android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
