# Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers

Source code for the paper *"Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers"* accepted at ICML 2025 ([arxiv](https://arxiv.org/abs/2403.07404)).

**Authors**: Filip Szatkowski, Yaoyue Zheng, Fei Yang, Bartłomiej Twardowski, Tomasz Trzciński, Joost van de Weijer

## 📄 Paper overview

Continual learning is crucial for applying machine learning in challenging, dynamic, and often resource-constrained environments. However, catastrophic forgetting - overwriting previously learned knowledge when new information is acquired - remains a major challenge. In this work, we examine the intermediate representations in neural network layers during continual learning and find that such representations are less prone to forgetting, highlighting their potential to accelerate computation. Motivated by these findings, we propose to use auxiliary classifiers(ACs) to enhance performance and demonstrate that integrating ACs into various continual learning methods consistently improves accuracy across diverse evaluation settings, yielding an average 10% relative gain. We also leverage the ACs to reduce the average cost of the inference by 10-60% without compromising accuracy, enabling the model to return the predictions before computing all the layers. Our approach provides a scalable and efficient solution for continual learning.


## ▶️ Reproducing the Results  

This repository builds on top of the [FACIL framework](https://github.com/mmasana/FACIL) and follows most of its conventions. To set it up, please install the dependencies listed in `requirements.txt`.  

We provide code to reproduce the experiments from the paper via Slurm grid searches, located in `scripts/slurm_grids`. These scripts can also be used to inspect the hyperparameters we explored during development. The grids call method-specific scripts from `scripts/templates`, whose naming scheme is straightforward: directories correspond to experimental settings, and filenames to the continual learning methods. We also provide plotting scripts in `src/plotting` that should enable at least partial reproduction of the paper’s figures.  

**Please note:**  
- Exact results may vary due to randomness in training and hardware differences.  
- Some shell scripts for running experimental grids and plotting may require small tweaks to run smoothly.  
- The main codebase has been kept as clean as possible, and we hope these supporting scripts provide a solid starting point for reproducing our results.

## 📚 Citing

If you use this code or base your work on our paper, please cite it as follows:

```bibtex
@inproceedings{szatkowskiimproving,
  title={Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers},
  author={Szatkowski, Filip and Zheng, Yaoyue and Yang, Fei and Trzcinski, Tomasz and Twardowski, Bart{\l}omiej and van de Weijer, Joost},
  booktitle={Forty-second International Conference on Machine Learning (ICML) 2025}
}
