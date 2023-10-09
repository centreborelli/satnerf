### create satnerf venv
conda create -n ba -c conda-forge python=3.8 libgdal
conda activate ba
python3 -m pip install --ignore-installed certifi gdal
python3 -m pip install --force-reinstall -v "setuptools<=58" pip
python3 -m pip install --ignore-installed git+https://github.com/centreborelli/sat-bundleadjust
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install fire
conda deactivate
echo "ba conda env created !"
