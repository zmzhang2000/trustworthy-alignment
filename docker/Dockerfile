FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo openssh-server rsync git vim htop zip unzip bzip2 curl wget screen tmux \
    && echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt install -y ttf-mscorefonts-installer \
    && apt-get -y autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN pip install flash-attn==2.3.3 datasets==2.14.6 sentencepiece==0.1.99 protobuf==3.20.3 accelerate==0.24.1 \
    tensorboard==2.15.1 transformers==4.35.0 deepspeed==0.12.4 openai==1.9.0 matplotlib==3.8.2 seaborn==0.13.2 \
    && pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@28ec7fa950346b5a895e85e1f3edd5648168acc4 \
    && rm -rf /root/.cache && rm -rf /tmp/*

# Remove ".weight" in line 48-51 of llama.py. This is a bug in deepspeed.
# Substitute parameter bias in line 156 of ds_attention.py. Otherwise may cause NaN. This is a bug in deepspeed.
RUN sed -i '48,51s/.weight//g' /opt/conda/lib/python3.10/site-packages/deepspeed/module_inject/containers/llama.py \
    && sed -i '156,156s/bias=self._attn_qkvb/bias=self._attn_qkvb if self.attn_qb is not None else None/g' /opt/conda/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/ds_attention.py

RUN git clone https://github.com/zmzhang2000/trustworthy-alignment.git && cd trustworthy-alignment && pip install -e .
