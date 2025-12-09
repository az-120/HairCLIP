# Project Setup

Follow these steps to get HairCLIP running locally. We recommend using our linked user interface, however.

## 1. Prerequisites
- Python 3.10+, `pip`, `virtualenv`

## 2. Git Setup and Environment
```bash
# First, fork the repository and clone it locally on to your device
git clone <your link>
cd HairCLIP

# create and activate a virtual environment (feel free to swap for conda)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## 3. Working with the Repo
1. Place images inside `data/test`.
2. To test scripts, from the repo root, run python3/<script_name>.py in the CLI.
3. run_inpainting.py generates the images. It requires cuda support, so if your local machine doesn't support that, you should use the `notebooks/hairclip_diffusion.ipynb` notebook in Google Colab.
4. In order to have local changes reflected in colab, you must git commit and push to your repo. After doing so, you can continue with the notebook.
5. In the notebook, follow the first 9 cells for image generation. Make sure to clone your own forked repository instead of this one, and choose your correct branch.
6. The 9th code cell will display out the generated images. Make sure you're using the proper range of prompts.
7. The following 3 cells are for running hyperparamter grid search, which runs `src\eval_runner.py`, but isn't needed for image generation.
8. Whenever you make more local changes, but the notebook is still running, all you have to do is run the git pull cell and the !python3 src/run_inpaintin.py cell to generate the images.
