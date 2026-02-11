tested on Raspberry Pi 4/5 miniconda

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