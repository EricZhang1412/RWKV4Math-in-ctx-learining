# In-Context Learning with RWKV-v7: A case study of simple function classes

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Inspired by the paper: **What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

![](setting.jpg)

``` bibtex
@InProceedings{garg2022what,
    title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
    author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
    year={2022},
    booktitle={arXiv preprint}
}
```
# 🎉[2025-6-19] Update:
- 🚩Implementation of RWKV-v7 training on simple function classes regression task:
  
  Model defined in [`src/models_rwkv_x070.py`](https://github.com/EricZhang1412/RWKV4Math-in-ctx-learining/blob/main/src/models_rwkv_x070.py) and training script in `src/train_rwkv.py`.
  > ⚠️This project is not based on Pytorch-lightning.
  >
  > ⚠️The core kernel cannot be supported by rwkv-fla based on Triton(e.g.`token_shift`, `fused_addcmul_rwkv7`, `fused_k_rwkv7`). So far, I have not found out why...
  

The instructions of the new hyperparameters are coming soon...

## 📝TODO
- [ ] Add training results and comparisons with other models. (wandb)
- [ ] Implementation of the RWKV-Deoth-recurrent model. (So HARRRRRD!!!!!!)
## Getting started
### Transformers (GPT-2)
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ``` shell
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ``` shell
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy.yaml` for a quick training run.

### RWKV-v7
1. train
    ``` shell
    cd src
    sh init_train_rwkv.sh # (stage 1, initialize the model)
    sh train_rwkv.sh # (stage 2, train the model)
    ```

## Instructions
### Baseline
In total, we have implemented several baselines focusing on different sets of problems, defined in [`src/models_rwkv_x070.py`](https://github.com/EricZhang1412/RWKV4Math-in-ctx-learining/blob/e03919041b2c58fad91c19108c78062d991b22c1/src/models_rwkv_x070.py#L38)
- `linear_regression`: $Y=A^\top X+B$
  - LeastSquaresModel
  - K-NN ($k = 3$)
  - AveragingModel
