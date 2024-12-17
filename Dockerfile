FROM nvidia/cuda:12.4.1-base-ubuntu20.04

ENV LANG=C.UTF-8 
ENV http_proxy=http://proxy.onera:80
ENV https_proxy=http://proxy.onera:80
ENV ftp_proxy=http://proxy.onera:80
ENV no_proxy=.onera
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries

RUN apt-get -y update --fix-missing && \
    apt-get -y install wget \
    curl \
    build-essential \
    git-core \
    python3-pip && \
    apt-get clean 
RUN rm -rf /var/lib/apt/lists/*


WORKDIR /home/dl_toolbox 
COPY requirements.txt ./
RUN pip3 --no-cache-dir install --upgrade -r requirements.txt 
COPY src ./src
VOLUME /home/dl_toolbox
EXPOSE 7745
RUN pip3 install jupyter

CMD HYDRA_FULL_ERROR=1 python3 src/train.py -m +experiment=rellis datamodule.data_path=./data trainer.limit_train_batches=1 trainer.limit_val_batches=1 trainer.limit_predict_batches=1 
#CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=7745

