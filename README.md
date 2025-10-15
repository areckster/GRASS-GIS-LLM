# GRASS-GIS-LLM Guide

A concise setup and training guide for working with the GRASS-GIS language model resources contained in this repository.

---

## 🖥️ Tested Environment
- **Operating System:** Ubuntu 24.04.3 LTS
- **CPU:** Intel® Xeon® W-2265 × 24
- **GPU:** NVIDIA Quadro RTX 4000
- **Python:** 3.12.3 (within a `venv` virtual environment)

> ⚠️ The steps below were validated in the environment above. Behaviour on other platforms may vary.

---

## 🚀 Quick Start Overview
1. [Set up Python and a virtual environment](#-set-up-the-python-environment)
2. [Install training dependencies](#-train-the-model)
3. [Generate fine-tuning data (optional)](#-create-training-data)

---

## 🧰 Set Up the Python Environment

Ensure Python is available:

```bash
python --version
python3 --version
```

Create and activate a virtual environment (replace `python3` with your Python executable if needed):

```bash
python3 -m venv venv
source venv/bin/activate
```

Deactivate the environment when finished:

```bash
deactivate
```

---

## 🤖 Train the Model

Install the required libraries:

```bash
pip install -U transformers accelerate peft bitsandbytes datasets huggingface_hub
```

Kick off training:

```bash
python train.py --data ./TRAINING_DATA_FILE_NAME.jsonl --out ./out-distilgptoss --epochs 2
```

This command downloads a distilled version of GPT-OSS 20B (distilled into Qwen 3 4B) and begins fine-tuning it using the provided dataset generated with `generate_grass_dataset.py`.

---

## 🗂️ Create Training Data

Install the scraper requirement:

```bash
pip install beautifulsoup4
```

1. Visit the GRASS GIS manuals: [https://grass.osgeo.org/learn/manuals/](https://grass.osgeo.org/learn/manuals/)
2. Download the latest documentation artifact. At the time of writing, the archive was available at:<br>
   `https://github.com/OSGeo/grass/actions/runs/18477726098/artifacts/4259709904`<br>
   You must be logged into GitHub to access the download.
3. Extract the downloaded ZIP archive.
4. Copy `generate_grass_dataset.py` into the extracted documentation folder so it sits alongside `grass_logo.txt` (one directory above the `addons` and `assets` folders).

Run the dataset generator:

```bash
python3 generate_grass_dataset.py --folder . --out grass_finetune_dataset.jsonl
```

The script produces approximately 1,000 training examples and writes them to `grass_finetune_dataset.jsonl` in the same directory.

> 📝 Dataset generation was only tested with GRASS 8.5.0dev documentation. Formatting in other releases may differ slightly, but the overall structure is expected to remain similar.

---

Happy mapping! 🌿
