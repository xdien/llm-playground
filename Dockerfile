# Step 1: Use the official ROCm development image as the base
FROM rocm/dev-ubuntu-22.04:6.3

# Step 2: Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Step 2: Add deadsnakes PPA and install Python 3.12 with dependencies
# The PPA is required because the default Ubuntu repos don't have python3.12.
# We also make sure git and other build tools are installed here.
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        rocrand-dev \
        hiprand-dev \
        rocblas-dev \
        hipsparse-dev \
        hipblas-dev \
        miopen-hip-dev \
        hipfft-dev \
        rccl-dev \
        hipcub-dev \
        rocthrust-dev \
        hipcub-dev \
        rocthrust-dev \
        hipsolver-dev \
        hipsparselt-dev \
        python3.12 \
        python3.12-dev \
        python3-setuptools \
        python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Install a modern version of CMake (v3.26+ is required by vLLM)
# The version from apt is too old, so we download and install it manually.
RUN CMAKE_VERSION=3.29.3 && \
    curl -L -O https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# Step 5: Update alternatives to make python3.12 the default python and python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Step 6: Install a modern version of pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Step 7: Set up working directory and environment variables
WORKDIR /app
ENV VLLM_TARGET_DEVICE=rocm
ENV PYTORCH_ROCM_ARCH=gfx906
ENV ROCM_PATH=/opt/rocm-6.3.0
ENV CUDA_HOME=$ROCM_PATH
ENV HUGGING_FACE_HOME=/huggingface_cache
ENV TRANSFORMERS_CACHE=/huggingface_cache
ENV HF_HOME=/huggingface_cache

# Step 8: Clone the vLLM repository
RUN git clone https://github.com/vllm-project/vllm.git .

# Step 9: Install PyTorch for ROCm 6.3 and dependencies
# Pin numpy to a compatible version before installing torch
# Install torch, torchvision, and torchaudio from the nightly builds for ROCm 6.3
RUN pip install numpy==1.26.4 && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3

# Step 10: Build and install vLLM from source
# We use --no-build-isolation to prevent pip from creating an isolated environment
# that doesn't have the pre-installed torch for ROCm.
# We also ensure setuptools, wheel, and setuptools_scm are up-to-date.
RUN pip install -U pip setuptools wheel setuptools_scm && \
    pip install --no-build-isolation --no-cache-dir -e .

# Step 11: Clean up apt cache
RUN rm -rf /var/lib/apt/lists/*

# Step 12: Copy your project files into the container
WORKDIR /app/project
COPY main.py .
COPY requirements.txt .

# Step 13: Install the remaining application-specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 14: Expose the port
EXPOSE 8000

# Step 15: Define the run command
CMD ["python", "main.py"] 