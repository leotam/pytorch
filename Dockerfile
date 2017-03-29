FROM nvcr.io/nvidia/torch:17.03
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN date
RUN apt-get update -y
RUN apt-get dist-upgrade -y
RUN apt-get -y install sudo lsb-release

RUN apt-get -y install build-essential software-properties-common
RUN apt-get -y install libhdf5-dev
RUN apt-get -y install g++ curl
RUN apt-get -y install libssl-dev apache2-utils libxml2-dev
RUN apt-get -y install sshfs
RUN apt-get -y install supervisor
RUN apt-get -y install tmux
RUN apt-get -y install git
RUN apt-get -y install lsof net-tools
RUN apt-get -y install vim vim-common

ENV CUDNN_VERSION 6.0.20 
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libcudnn6=$CUDNN_VERSION-1+cuda8.0 \             
         libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py35 python=3.5.2 numpy scipy ipython mkl&& \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/envs/pytorch-py35/bin:$PATH
RUN conda install --name pytorch-py35 -c soumith magma-cuda80
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .

RUN cat requirements.txt | xargs -n1 pip install --no-cache-dir && \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_LIBRARY_PATH=/opt/conda/envs/pytorch-py35/lib \
    CMAKE_INCLUDE_PATH=/opt/conda/envs/pytorch-py35/include \
    pip install -v .

WORKDIR /workspace
RUN chmod -R a+w /workspace
