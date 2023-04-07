# An asymmetric heuristic for trained ternary quantization based on the weights' statistics

## I) Introduction

GitHub repository for the sumbission of the paper *An asymmetric heuristic for trained ternary quantization based on the weights' statistics* for Machine Learning for Healthcare (MLHJ) 2023.

## II) Configuration

To be able to run the different codes (Linux platforms), you need to start by running the following command:

    export PYTHONPATH="${PYTHONPATH}:pathToThe_aTTQ_Code"

Then, you should install the different libraries needed to execute the different codes:

    pip install -r requirements.txt

## III) Proposed method

![image](https://github.com/attq-submission/aTTQ/blob/main/figs/MethodOverview.jpg) 

In a nutshell, our proposed method is composed of three steps, inspired from TTQ ([Zhu et al. (2016)](https://arxiv.org/abs/1612.01064)):
- **Pruning:** pruning is done before ternarization based on the weights's statistics, by introducing two asymmetric parameters controlling the sparsity rate.
- **Ternarization**: the remaining positive weights are set to $1$ and the negatives ones to $-1$.
- **Scaling**: two full-precision scaling trainable parameters are assocaited to the ternary weights tensor, one for the positive weights $W_r$, and one for the negative ones, $W_l$.

## IV) Code structure

TODO

## V) Examples

TODO


