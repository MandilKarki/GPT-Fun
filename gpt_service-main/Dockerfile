FROM tensorflow/tensorflow:2.5.0
RUN apt update && \
    apt-get install git -y

WORKDIR /app/
COPY . /app/
RUN apt install -y zstd wget

RUN git clone https://github.com/kingoflolz/mesh-transformer-jax && \
    pip install -r mesh-transformer-jax/requirements.txt && \
    pip install mesh-transformer-jax/ jax==0.2.12 jaxlib==0.1.67 &&  \
    pip install fastapi uvicorn requests aiofiles aiohttp

RUN pip install  -r requirements.txt

ENV PYTHONPATH /app:/app/mesh-transformer-jax:/usr/local/bin/python3
ENV PATH $PYTHONPATH:$PATH
ENV TOKENIZERS_PARALLELISM=true
EXPOSE 8080

RUN chmod +x ./start_api.sh
CMD ["./start_api.sh", "8080"]
