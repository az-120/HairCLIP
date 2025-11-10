# Project Setup

Follow these steps to get HairCLIP running locally for grading or development. Replace placeholders (❗) with the exact commands or credential notes for your final submission.

## 1. Prerequisites
- Python 3.10+ and `pip`
- (Optional) `conda` or `virtualenv`
- Git LFS if you plan to sync large assets such as trained models or demo videos

## 2. Clone and Environment
```bash
# clone the repository
❗ git clone <your-repo-url>
cd HairCLIP

# create and activate a virtual environment (feel free to swap for conda)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## 3. Data and Models
1. Place raw datasets inside `data/`. Document exact filenames and download links here.
2. Put pretrained weights or checkpoints inside `models/`. Include the expected directory layout, e.g. `models/hairclip.ckpt`.
3. If your datasets or models require authentication keys or license acceptance, describe that flow explicitly so graders can request access.

## 4. Running the Project
- Core scripts live in `src/`. Provide the main entry point command, e.g. `python src/app.py --input path/to/image.jpg`.
- Jupyter exploration notebooks belong in `notebooks/`. Add clear instructions for any notebook that must be executed for evaluation.
- Document any environment variables or API keys (e.g. `OPENAI_API_KEY`) needed to hit external services, including whether graders need their own keys or if you provide mock responses.

## 5. Testing & Evaluation
Explain how graders can reproduce your reported metrics:
```bash
# example
pytest
python src/eval.py --config configs/base.yaml
```
Add information about deterministic seeds, hardware expectations (GPU/CPU), and runtime (approximate minutes) to set expectations.

## 6. Troubleshooting
- If dependency resolution fails, delete `.venv` and recreate it from scratch.
- For CUDA issues, provide the exact PyTorch + CUDA versions you validated.
- Capture common runtime errors and their fixes as you discover them.
