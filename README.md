# Pink Edge AI

> Offline-first medical-imaging triage for resource-constrained healthcare settings.

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://pinkedgeai-zhuanifcsfpchwryujb28z.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Not%20specified-lightgrey)](#license)

Pink Edge AI is a research and hackathon prototype that brings medical-image triage workflows to a local edge computer. It provides a Tkinter desktop application and a responsive Streamlit web application over shared Python inference, reporting, and local-cache logic.

The project is designed around a practical constraint: a Basic Health Unit may have limited connectivity, limited compute, and no radiologist on site. The application can process images locally, save reports to SQLite, and present a simulated escalation and synchronization workflow for review.

**This project is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.** All predictions require review by a qualified healthcare professional.

## Live Demo

Open the deployed Streamlit application:

<https://pinkedgeai-zhuanifcsfpchwryujb28z.streamlit.app/>

The public demo may take time to wake up or install its machine-learning dependencies. For reproducible development, run the application locally using the instructions below.

## Capabilities

- Three modality workflows: mammography, tuberculosis chest X-ray, and maternal ultrasound.
- Offline-first inference after model weights and Python dependencies are available locally.
- English and Urdu interface options.
- Upload JPEG and PNG scans or use generated placeholder images for UI testing.
- Triage result cards with confidence, severity, localization, and escalation status.
- Local SQLite report cache for offline operation.
- Downloadable text and PDF reports.
- Simulated GSM failover, hospital-hub alerts, cloud synchronization, OSS backup, and OTA status panels.
- Desktop UI through Tkinter and browser UI through Streamlit.

## Model Status

The application reports whether a modality is backed by a real model or a clearly labeled simulation fallback.

| Modality | Current behavior | Model or source |
| --- | --- | --- |
| Tuberculosis | Real local inference | YOLOv8 classification checkpoint trained on the TBX11K simplified dataset; classes are `no_tb` and `tb` |
| Maternal health | Real inference when the public checkpoint downloads successfully | `shr3m/fetal-brain-plane-cnn` from Hugging Face; fetal-brain ultrasound plane classification |
| Mammography | Simulated by default | Optional Roboflow export when `ROBOFLOW_API_KEY` is configured; otherwise the UI labels the result as simulated |

The committed TB checkpoint is located at `Misc/Pink_Edge_AI-main/models/tb_classifier.pt`. `inference.py` searches this location as well as the conventional `models/` and `Models/` directories so the application works on case-sensitive deployment systems such as Linux-based Streamlit Cloud.

Detailed model provenance, preprocessing, licenses, and limitations are documented in [Documentations/MODEL_SOURCES.md](Documentations/MODEL_SOURCES.md).

## Architecture

```text
                     +-----------------------------+
                     |  Tkinter desktop UI (GUI.py)|
                     +--------------+--------------+
                                    |
                     +--------------v--------------+
                     | Shared triage and cache code |
                     | GUI.py + inference.py       |
                     +------+-----------------------+
                            |
              +-------------+-------------+
              |                           |
   +----------v----------+      +---------v----------+
   | Streamlit web UI    |      | Local model files  |
   | streamlit_app.py    |      | YOLO / PyTorch     |
   +----------+----------+      +---------+----------+
              |                           |
              +-------------+-------------+
                            |
                   +--------v--------+
                   | SQLite cache    |
                   | Reports / sync  |
                   +-----------------+
```

The cloud-sync and GSM panels are workflow simulations for the prototype. They do not require Alibaba Cloud credentials and do not provide a production messaging or storage integration.

## Technology Stack

- Python 3.10 or newer
- Streamlit for the responsive web application
- Tkinter for the desktop application
- PyTorch and Torchvision for neural-network inference
- Ultralytics YOLO for the trained TB classifier and optional mammography model
- OpenCV, NumPy, and Pillow for image processing
- SQLite for local report storage
- FPDF2 for PDF reports
- Hugging Face Hub for optional public model downloads
- Roboflow SDK for optional mammography weights

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/Zobia-Irshad/Pink_Edge_AI.git
cd Pink_Edge_AI

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the project with the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
git clone https://github.com/Zobia-Irshad/Pink_Edge_AI.git
cd Pink_Edge_AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch, Ultralytics, and OpenCV packages can make the first installation large. A network connection is required to install dependencies and download any public model that is not already present locally.

## Run the Applications

### Streamlit web application

```bash
streamlit run streamlit_app.py
```

Open <http://localhost:8501>. On Windows, `Start_Web.bat` installs missing dependencies and launches the same entry point.

To make the local server reachable from another device on the same network:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Tkinter desktop application

```bash
python GUI.py
```

On Windows, `Start.bat` installs missing dependencies and launches the desktop application.

## Using the Application

1. Choose a modality from the sidebar.
2. Upload a JPEG or PNG scan, or run the workflow with a generated placeholder image.
3. Select **Run Triage**.
4. Review the verdict, confidence, model source, severity, and localization fields.
5. Save the report to the local cache if required.
6. Generate a text or PDF report from the dashboard.
7. Use GSM Failover mode to inspect the prototype alert and synchronization workflow.

Generated placeholders are for interface testing only. For model testing, use representative images from the appropriate modality and treat every result as research output.

## Optional Mammography Configuration

Mammography uses the explicit simulated fallback unless a trained Roboflow export is available. To enable the optional path:

1. Create a Roboflow API key for the configured project.
2. Set the key as an environment variable:

   ```powershell
   $env:ROBOFLOW_API_KEY = "your-key"
   ```

   Or save the key in `roboflow_key.txt` beside `GUI.py`. Do not commit this file.
3. Restart the application.

The loader keeps the simulated fallback if the project has no downloadable trained export or if the request fails. It never treats a missing model as a real clinical prediction.

## Training the TB Classifier

The training script uses the TBX11K simplified archive and creates a YOLO classification dataset with `no_tb` and `tb` folders. The dataset and generated runs are intentionally ignored by Git because they are large and contain source images.

Expected archive location:

```text
dataset/tbx11k-simplified.zip
```

Run training with:

```bash
python train_tb_classifier.py
```

The script extracts the archive, verifies that image files are readable, creates a stratified train/validation split, trains a CPU-friendly YOLOv8 classifier, and copies the best checkpoint to the application model path. Training settings can be adjusted in `train_tb_classifier.py` for a larger dataset, more epochs, or GPU execution.

The repository includes the trained TB checkpoint, but not the full TBX11K dataset. Confirm that your use of the dataset and resulting weights complies with the dataset's terms before redistribution or deployment.

## Validation

Run the repository validation suite from any working directory:

```bash
python Validation/validate.py
```

The suite covers imports, image generation, overlay drawing, simulated fallbacks, SQLite cache round trips, report generation, real-model inference, triage dispatch, desktop UI construction, and a headless Streamlit smoke test. It exits with a non-zero status when a check fails.

For a focused TB model smoke test:

```bash
python -c "from pathlib import Path; from PIL import Image; import inference; print(inference.predict_tb(Image.open(next(Path('Test Data/Tuberculosis').glob('*')))))"
```

## Streamlit Community Cloud Deployment

1. Push the repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app from the `main` branch.
4. Set the main file to `streamlit_app.py`.
5. Deploy.

The top-level `requirements.txt` is used by Streamlit Cloud. The first boot can be slow because PyTorch, Ultralytics, and model files are larger than typical Streamlit dependencies. The free tier may not have enough memory for every model to load simultaneously; use a larger deployment tier or disable optional model downloads if necessary.

For the optional mammography model, configure the secret in the Streamlit app settings:

```toml
ROBOFLOW_API_KEY = "your-key"
```

Never commit API keys, patient images, generated databases, virtual environments, or downloaded datasets.

## Repository Layout

```text
GUI.py                         Tkinter desktop application and shared triage logic
streamlit_app.py               Streamlit web application
inference.py                   Lazy model loaders and modality predictors
train_tb_classifier.py         TBX11K preparation and YOLOv8 training script
requirements.txt               Shared Python dependencies
Start.bat                      Windows desktop launcher
Start_Web.bat                  Windows Streamlit launcher
Validation/validate.py         Automated validation suite
Test Data/                     Small local sample images for validation
Models/                        Downloaded model cache, including maternal weights
Misc/Pink_Edge_AI-main/        Original project materials and trained TB checkpoint
Documentations/                Architecture, model, backend, frontend, and user docs
Assets/                        Change notes and supporting project assets
```

The original notebook and hackathon submission are retained under `Misc/` for reference. The maintained application entry points are `GUI.py` and `streamlit_app.py`.

## Privacy and Safety

- Images are processed locally by default.
- The local SQLite database is not encrypted and should be treated as sensitive.
- Do not use real patient data in development unless you have the required authorization and safeguards.
- Do not upload patient images or reports to public issue trackers, demo sites, or untrusted storage.
- Confidence values, severity labels, localization fields, and escalation messages are prototype outputs, not clinical measurements.
- A qualified clinician must review all results before any healthcare action.

## Documentation

- [Model sources and licenses](Documentations/MODEL_SOURCES.md)
- [Project architecture](Documentations/PROJECT_ARCHITECTURE.md)
- [Backend documentation](Documentations/BACKEND_DOCUMENTATION%20(2).md)
- [Frontend documentation](Documentations/FRONTEND_DOCUMENTATION%20(1).md)
- [API documentation](Documentations/API_DOCUMENTATION.md)
- [User guide](Documentations/USER_GUIDE.md)
- [Change history](Assets/Changes/Changes.md)

## Relationship to the Original Submission

`Misc/Pink_Edge_AI-main` contains the original Alibaba Cloud hackathon submission. The maintained application in this repository preserves its clinical vocabulary, report concepts, and edge-first intent while providing a responsive Streamlit rebuild, shared model loaders, a real trained TB classifier, and explicit simulated fallbacks where a production-ready model is not available.

## License

This repository does not currently include a root `LICENSE` file. The project code and bundled assets should not be redistributed as an open-source package until a project license is added. Third-party model checkpoints, datasets, images, and documentation remain subject to their own licenses and terms; see [Documentations/MODEL_SOURCES.md](Documentations/MODEL_SOURCES.md) before reuse.

## Acknowledgements

- Streamlit and the open-source Python scientific-computing ecosystem.
- Ultralytics YOLO and PyTorch.
- The TBX11K dataset used for the trained TB classifier.
- The public Hugging Face fetal-brain-plane checkpoint used for maternal ultrasound classification.
- The original Pink Edge AI Alibaba Cloud hackathon team and submission preserved under `Misc/`.
