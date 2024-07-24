print('='*30)
print(f'{"Library versions": ^30}')
print('='*30)

# cmd: python scripts/verify_env.py  (self-made)
# cmd: python -m mmdet3d.utils.collect_env (API)

# ---------------------------
# 0 - Python (3.8.16)
# ---------------------------
# Not working (mim package problem):
# conda create --name redformer python=3.8.16
# conda activate redformer
# conda env update --name redformer --file environment.yml 

# Todo: Create a bash file to install and other to verify all

# (optional) clone env: 
# conda create --name new_env --clone original_env

# ---------------------------
# 1 - Pytorch (1.9.1 + CUDA 11.1)
# ---------------------------
# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

# pip install torch==1.9.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
try: 
    import torch
    import torchvision
    import torchaudio
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f'torch=={torch.__version__} (cuda {torch.version.cuda})')
    print(f" ->used_device={device}")
    print(f" ->cudnn.enabled={torch.backends.cudnn.enabled}")
    print(f" ->cudnn.version={torch.backends.cudnn.version()}")
    print(f'torchvision=={torchvision.__version__}')
    print(f'torchaudio=={torchaudio.__version__}\n')
except: print('torch==(error)\n')

# pip install nuscenes-devkit==1.1.10
# try: 
#     import nuscenes-devkit
#     print(f'nuscenes-devkit=={nuscenes-devkit.__version__}')
# except: print('nuscenes-devkit==(not installed)')

# ---------------------------
# 2 - MMlab
# ---------------------------
# pip install openmim

# ------------
# 2.1 -> mmcv-full (1.6.0)
# ------------
# mim install mmcv-full==1.6.0
try: 
    import mmcv
    print(f'mmcv=={mmcv.__version__}')
except: print('mmcv==(not installed)')

# ------------
# 2.2 -> mmdet (2.28.2)
# ------------
# mim install mmdet==2.28.2
try: 
    import mmdet
    print(f'mmdet=={mmdet.__version__}')
except: print('mmdet==(not installed)')

# ------------
# 2.3 -> mmseg (0.30.0)
# ------------
# mim install mmsegmentation==0.30.0
try: 
    import mmseg
    print(f'mmseg=={mmseg.__version__}')
except: print('mmseg==(not installed)')

# ------------
# 2.4 -> mmdet3d (1.0.0rc6)
# ------------
# cd ../..
# git clone -b 1.0 https://github.com/open-mmlab/mmdetection3d.git
# cd mmdetection3d
# pip install -v -e . # or python setup.py develop

# (optional)
# verify all branchs:  git branch -a
# change branch:       git checkout v1.0
# install new version: pip install -v -e . # or python setup.py develop
try: 
    import mmdet3d
    print(f'mmdet3d=={mmdet3d.__version__}\n')
except: print('mmdet3d==(not installed)\n')

# pip install numpy==1.23.5
try: 
    import numpy
    print(f'numpy=={numpy.__version__}')
except: print('numpy==(not installed)')

# conda install matplotlib==3.5
try: 
    import matplotlib
    print(f'matplotlib=={matplotlib.__version__}')
except: print('matplotlib==(not installed)')

# pip install setuptools==59.5.0
try: 
    import setuptools
    print(f'setuptools=={setuptools.__version__}')
except: print('setuptools==(not installed)')

# pip install numba==0.53.0
try: 
    import numba
    print(f'numba=={numba.__version__}')
except: print('numba==(not installed)')

print()

# Errors:
# 1: 
'''
ImportError: /home/lasi/anaconda3/envs/redformer2/lib/python3.8/site-packages/mmcv/_ext.cpython-38-x86_64-linux-gnu.so: 
undefined symbol: _ZNK3c1010TensorImpl36is_contiguous_nondefault_policy_implENS_12MemoryFormatE

ans: pip install --upgrade mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA}/torch${PYTORCH}/index.html
where you should replace or set "${MMCV}", "${CUDA}", and "${PYTORCH}" with their versions. Also, "${CUDA}" is in 101 for CUDA 10.1 and so on. 
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html

source:[https://github.com/open-mmlab/mmdetection/issues/4291]
'''