<div align="center">

# journeycv

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

A data provenance framework for building better deep learning computer vision projects <br>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<br>


<br>
  
Click on [<kbd>Use this template</kbd>](https://github.com/deepseek-eoghan/journeycv/generate) to initialize new repository.

<br>
  
Check out the sample wandb dashboard for this run here <a> https://wandb.ai/eoghan/journeycv?workspace=user-eoghan </a>
  
</div>

# Introduction
<p>
In computer vision it is fair to say that the journey (experimental process) is just as important as the destination (trained model). There are various input and output components which need to be tracked in order to ensure provenance such as hyperparameters, input data and output test metrics. While many deep learning practitioners will likely version control their model architecture through a familiar text based tool such as git or svn, the process for tracking the hyperparameters, dataset, generated model weights and test metrics etc. cannot always be achieved easily using the same methodologies.
</p>

![image](https://user-images.githubusercontent.com/82596496/156372962-e915a6ea-f7bf-460d-9331-d4593c1ab93c.png)

# Getting Started

Follow these steps to set up the repository

```
# clone the project
git clone https://github.com/deepseek-eoghan/journeycv
cd journeycv

conda create -n journeycv python=3.8
conda activate journeycv

conda install -c anaconda cython
pip install -r requirements.txt
```
