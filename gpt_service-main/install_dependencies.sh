#!/usr/bin/env bash
mkdir /tmp/gpt
cd /tmp/gpt || exit
sudo apt install zstd
git clone https://github.com/kingoflolz/mesh-transformer-jax
pip install -r mesh-transformer-jax/requirements.txt
pip install mesh-transformer-jax/ jax[tpu]==0.2.12 jaxlib==0.1.67
pip install fastapi uvicorn requests aiofiles aiohttp
