### create satnerf venv
conda create -n ba -c conda-forge python=3.8
conda activate ba
python3 -m pip install --force-reinstall -v "setuptools<=58"
python3 -m pip install git+https://github.com/centreborelli/sat-bundleadjust
pip install fire
#pip install "pyproj>=2.0.2,<3.0.0"
conda deactivate
echo "ba conda env created !"
