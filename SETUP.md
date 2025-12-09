# Project Setup

Follow these steps to get HairCLIP running locally. We recommend using our linked user interface, however.

## 1. Prerequisites
- Python 3.10+ and `pip`
- (Optional) `virtualenv`

## 2. Clone and Environment
```bash
# fork the repository
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
3. Notebooks should be ran separately in colab for cuda support, and image generation is guidelined in the README.
