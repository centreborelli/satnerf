### create s2p venv
conda create -n s2p -c conda-forge python=3.6 libgdal
conda activate s2p
pip install s2p gdal fire shapely
conda deactivate
echo "s2p conda env created !"
