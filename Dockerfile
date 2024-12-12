FROM nvidia/cuda:12.4.1-base-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        
