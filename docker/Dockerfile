# Init from
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
 
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
 
# Install dependencies
RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
RUN apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
 
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
 
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge
 
# Install mmcv
RUN pip install openmim==0.3.9
RUN mim install mmcv-full==1.6.0
RUN mim install mmdet==2.28.2
RUN mim install mmsegmentation==0.30.0
RUN mim install mmdet3d==1.0.0rc6
 
# Install other code dependences
RUN pip install nuscenes-devkit==1.1.10
RUN pip install tqdm==4.65.2
RUN pip install numpy==1.23.5
RUN pip install matplotlib==3.5
RUN pip install setuptools==59.5.0
RUN pip install numba==0.53.0
RUN pip install yapf==0.40.1
 
WORKDIR /code
 
EXPOSE 8080