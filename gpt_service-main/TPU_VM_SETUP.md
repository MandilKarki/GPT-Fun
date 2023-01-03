## tpu-vm setup

gcloud config set account your-email-account
gcloud config set project project-id

gcloud services enable tpu.googleapis.com

gcloud beta services identity create --service tpu.googleapis.com

gcloud config set compute/zone us-central1-f

gcloud alpha compute tpus tpu-vm create treeleaf-tpu-1 \
--zone us-central1-f \
--accelerator-type v2-8 \
--version v2-alpha

gcloud alpha compute tpus tpu-vm ssh treeleaf-tpu-1 --zone us-central1-f

sudo mkdir /app
sudo chown -R $(id -u):$(id -g) /app
https://github.com/treeleaftech/gpt_service.git

 sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo docker-compose up -d


## References
---
https://cloud.google.com/tpu/docs/users-guide-tpu-vm
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
https://github.com/kingoflolz/mesh-transformer-jax/tree/master/docker
