# Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning

Official repository for [Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning](https://openreview.net/forum?id=XwnABAdH5y)

## Installation

We provide two ways to install the running environment: `Docker` and `Conda`. Installing by `Docker` is highly recommend for reproducibility as the code is tested in the Docker container.

### Docker
First, make sure the Docker engine is running on your machine.

Then you can pull the pre-built image from Docker Hub by running the following command:
```shell
docker pull zmzhang2000/trustworthy-alignment:latest
```
Optionally, you can build the image from the Dockerfile:
```shell
cd docker
docker build -t zmzhang2000/trustworthy-alignment .
```

Finally, run the Docker container by:
```shell
docker run -it zmzhang2000/trustworthy-alignment
```

### Conda
If you prefer to install the environment manually, please follow the instructions below.
```shell
conda create -n base python=3.10.13 
conda activate base
conda install pytorch==2.1.0 -c pytorch
pip install -r requirements.txt
pip install -e .
```

Besides, if you install the running environment by Conda, you need to modify the following files to make the code run correctly, where `$CONDA_HOME` is the path to the Conda environment.
```shell
sed -i '48,51s/.weight//g' $CONDA_HOME/lib/python3.10/site-packages/deepspeed/module_inject/containers/llama.py
sed -i '156,156s/bias=self._attn_qkvb/bias=self._attn_qkvb if self.attn_qb is not None else None/g' $CONDA_HOME/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/ds_attention.py
```

## Data & Model Preparation

### Data

Download the [preprocessed data](https://drive.google.com/drive/folders/1b86HdJLuaz2mJoZkJARmkCs_m7rIImCj?usp=sharing) and put them under the `data` folder.

You can also preprocess the data by yourself following instructions in `data/nq`.

### Model

Request access and download the `Llama-2-7b-chat` model from [here](https://llama.meta.com/llama2/). Then put the model under the `ckpt/huggingface/meta-llama` folder.

## Training
To train the model, you can run the following command:
```shell
cd training
bash run_llama2_7b_lora.sh
```

## Evaluation

To evaluate the model, you can run the following command:
```shell
cd inference
bash evaluate.sh
```

## Reference
Please cite the following paper if Trustworthy-Alignment is helpful for your research
```
@inproceedings{zhang-etal-2024-trustworthy,
    title = "Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning",
    author = "Zhang, Zongmeng  and
      Shi, Yufeng  and
      Zhu, Jinhua  and
      Zhou, Wengang  and
      Qi, Xiang  and
      Zhang, Peng and
      Li, Houqiang",
    booktitle = "Proceedings of the 41st International Conference on Machine Learning",
    month = jul,
    year = "2024",
    address = "Vienna, Austria",
    publisher = "PMLR"
}
```