Grass LLM ProjectBelow you will find how to install and run all files contained in this repository.At the very bottom there are instructions as to how training data was created. This was tested on Ubuntu 24.04.3 LTS with an Intel® Xeon® W-2265 × 24 CPU and an Nvidia Quadro RTX 4000 GPU running Python 3.12.3 in a virtual environment via "venv"




Setup:
ensure python is installed
python --version
python3 --version

assuming python3 is correct, create a virtual environment:
python3 -m venv venv #creates venv
source venv/bin/activate #activates venv

deactivate with:
deactivate



Train:

install dependencies:
pip install -U transformers accelerate peft bitsandbytes datasets huggingface_hub

run:
python train.py --data ./TRAINING_DATA_FILE_NAME.jsonl --out ./out-distilgptoss --epochs 2

This downloads a distilled version of GPT-OSS 20b distilled into qwen3 4b and begins training on our training data generated with "generate_grass_dataset.py"





Training data creation:
Install requirements:
pip install beautifulsoup4

First, visit https://grass.osgeo.org/learn/manuals/
As of creating this, the latest release for the docs was found at this URL: https://github.com/OSGeo/grass/actions/runs/18477726098/artifacts/4259709904
Only tested on GRASS 8.5.0dev, though the formatting should most likely remain consistent among releases??????
We used this link:
https://github.com/OSGeo/grass/actions/runs/18477726098
And downloaded mkdocs-site (you will need to be logged in to GitHub in order to click it)

Download the zip
Unzip
Place generate_grass_data_set.py inside the newly created folder. It should be at the same level as "grass_logo.txt" in a dir higher than the addons folder assets folder etc. 


Run:
python3 generate_grass_dataset.py --folder . --out grass_finetune_dataset.jsonl
This will produce roughly 1000 examples of training data contained in a file named grass_finetune_dataset.jsonl in the same directory as the python script
