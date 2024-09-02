# Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning

Official repository for [Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning](https://proceedings.mlr.press/v235/zhang24bg.html)

## Installation

We provide two ways to install the running environment: `Docker` and `Conda`. Installing by `Docker` is highly recommend for reproducibility.

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
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

Besides, if you install the running environment by Conda, you need to modify the following files to make the code run correctly, where `$CONDA_HOME` is the path to the Conda environment.
```shell
CONDA_HOME=$(dirname $(dirname `which python`))
sed -i '48,51s/.weight//g' $CONDA_HOME/lib/python3.10/site-packages/deepspeed/module_inject/containers/llama.py
sed -i '156,156s/bias=self._attn_qkvb/bias=self._attn_qkvb if self.attn_qb is not None else None/g' $CONDA_HOME/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/ds_attention.py
```

## Data & Model Preparation

### Data

Download the [preprocessed data](https://drive.google.com/drive/folders/1b86HdJLuaz2mJoZkJARmkCs_m7rIImCj?usp=sharing) and put them under the `data` folder.

You can also preprocess the data by yourself following instructions in `data/nq`.

### Model

Request access to `meta-llama/Llama-2-7b-chat-hf` on [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). You can also train on other models by changing the `ACTOR_MODEL_PATH` in `training/run_llama2_7b_lora.sh`.

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

## Acknowledgements
This project is based on the [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) repository. We thank the authors for their great work.

## Citation
Please cite the following paper if Trustworthy-Alignment is helpful for your research
```
@InProceedings{pmlr-v235-zhang24bg,
    title = 	 {Trustworthy Alignment of Retrieval-Augmented Large Language Models via Reinforcement Learning},
    author =       {Zhang, Zongmeng and Shi, Yufeng and Zhu, Jinhua and Zhou, Wengang and Qi, Xiang and Zhang, Peng and Li, Houqiang},
    booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
    pages = 	 {59827--59850},
    year = 	 {2024},
    editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
    volume = 	 {235},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {21--27 Jul},
    publisher =    {PMLR},
    pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhang24bg/zhang24bg.pdf},
    url = 	 {https://proceedings.mlr.press/v235/zhang24bg.html}
}

```