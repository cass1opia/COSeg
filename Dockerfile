FROM --platform=linux/amd64 nvidia/cuda:11.3.1-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential cmake git \
    openssh-server sudo \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/sshd
RUN ssh-keygen -A

WORKDIR /workspace

# Set environment variables for CUDA compilation
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
# ENV FORCE_CUDA="1"

# COPY requirements.txt .
# # Install packages individually
# RUN pip3 install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip3 install --no-cache-dir numpy==1.20.3
# # Install torch_points3d without dependencies to avoid open3d conflicts
# RUN pip3 install --no-cache-dir --no-deps torch_points3d==1.3.0
# # Install open3d without dependencies to avoid jupyterlab conflicts
# RUN pip3 install --no-cache-dir --no-deps open3d==0.10.0.0
# # Install torch extensions that need compilation
# RUN pip3 install --no-cache-dir torch-scatter==2.1.1
# RUN pip3 install --no-cache-dir torch-points-kernels==0.6.10
# RUN pip3 install --no-cache-dir torch-geometric==1.7.2
# RUN pip3 install --no-cache-dir timm==0.9.2
# RUN pip3 install --no-cache-dir tensorboardX==2.6

# COPY . .
# RUN cd lib/pointops2 && python3 setup.py install


CMD ["/bin/bash"]