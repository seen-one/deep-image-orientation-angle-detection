## Raspberry Pi 4/5 miniconda

quickly dropping my notes here…

conda create -n tf python=3.9 -y

(outside of the environment)

conda install -c conda-forge pillow -y

remove Pillow from requirements.txt

pip install -r requirements.txt --no-deps

pip install numpy==1.26.4

pip install py360convert==1.0.4

python -m pip install torch torchvision torchaudio

conda activate tf

cd deep-image-orientation-angle-detection

python run.py


## Windows

conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install tensorflow==2.8.0
pip install numpy==1.23.0
pip install protobuf==3.20.*
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113   

pip install -r requirements.txt

pip install Flask==2.0.3 Werkzeug==2.0.3 Jinja2==3.0.3
pip install py360convert=1.0.4

## WSL Ubuntu
Ubuntu 22.04.5 LTS WSL

sudo nano /etc/apt/sources.list
change to local mirror



git clone https://github.com/pidahbus/deep-image-orientation-angle-detection.git
git clone https://github.com/seen-one/deep-image-orientation-angle-detection -b 2nd-pass

cd deep-image-orientation-angle-detection

sudo apt update && sudo apt upgrade -y
apt install python3.10-venv

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install "transformers[tensorflow]==4.41.2" "tensorflow[and-cuda]"

pip install flask pillow loguru matplotlib opencv-python transformers datasets scikit-learn pandas tqdm tf-keras py360convert

echo 'export TF_USE_LEGACY_KERAS=1' >> venv/bin/activate
export TF_USE_LEGACY_KERAS=1