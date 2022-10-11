### create satnerf venv
conda create -n satnerf -c conda-forge python=3.6 libgdal
conda activate satnerf
pip install --ignore-installed certifi -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate
echo "satnerf conda env created !"
