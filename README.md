tested on Raspberry Pi 4/5 miniconda (and Windows but need to re-confirm dependencies)

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