# Build Command:
#  $ cd ~/monarch
#  $ export TAG_NAME=$USER-dev
#  $ docker build --network=host -t monarch:$TAG_NAME -f Dockerfile .
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Configure Rust to use http proxy
RUN mkdir -p /root/.cargo && \
    cat <<EOF > /root/.cargo/config
[http]
proxy = "${http_proxy}"
[https]
proxy = "${https_proxy}"
EOF

# Install native dependencies
RUN apt-get update -y && \
    apt-get -y install curl clang libunwind-dev

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /monarch

# Install Python deps as a separate layer to avoid rebuilding if deps do not change
# TODO extract deps (currently in pyproject.toml and setup.py) into requirements.txt
# and uncomment the line below in favor of generating requirements.txt
# COPY requirements.txt .
RUN cat <<EOF > requirements.txt
# project-deps (from pyproject.toml and setup.py)
torch
pyzmq
requests
numpy
pyre-extensions
pytest-timeout
cloudpickle
pytest-asyncio
# build-system deps (from pyproject.toml)
setuptools
setuptools-rust
EOF

RUN pip install -r requirements.txt

# Build and install monarch
COPY . /monarch
RUN cargo install --path monarch_hyperactor
RUN python setup.py install
