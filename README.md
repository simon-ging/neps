# Comprehensive NAS
This repository contains code for different NAS (+ HPO) optimizers.

##  Prerequisites and Dependencies
This code repository contains the submission of NAS-BOWL. To run the codes, please see the prerequisites below:
1. Create a Python/Conda environment (we used Anaconda with Python 3.7). After that install other dependencies via ```poetry install```. Then install ```torch``` via ```python -m install_dev_utils.install_torch```
2. To run on tabular/surrogate benchmarks download the datasets, e.g., the NAS-Bench-201 datasets. We expect these files:

    For NAS-Bench-201 we expect: ```NAS-Bench-201-v1_1-096897.pth```

    Also install the relevant APIs.

## Experiments
To reproduce our exemplary experiments, see below

1. Search on NAS-Bench-201 (by default on the CIFAR-10 valid dataset.)
    ```bash
    python -u examples/nasbench_optimization.py  --dataset nasbench201 --task cifar10-valid --pool_size 200 --mutate_size 200 --batch_size 5 --n_init 10 --max_iters 30 --log --optimize_arch
    ```
    Append ```--fixed_query_seed 3``` for deterministic objective function. Append ```--task cifar100```
    for CIFAR-100 dataset, and similarly ```--task ImageNet16-120``` for ImageNet16 dataset.


## References
We used materials from the following public code repositories. We would like to express our gratitude towards
these authors/codebase maintainers

   1. Ru, Binxin al. "Interpretable Neural Architecture Search via Bayesian Optimisation with Weisfeiler-Lehman Kernels."
   International Conference on Machine Learning (2021). [https://github.com/xingchenwan/nasbowl]
