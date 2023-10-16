# $f$-Policy Gradients: A General Framework for Goal Conditioned RL using $f$-Divergences

This repository is the source code for the paper [$f$-Policy Gradients: A General Framework for Goal Conditioned RL using $f$-Divergences](https://arxiv.org/abs/2310.06794v1) published in the Thirty Seventh Conference on Neural Information Processing Systems (NeurIPS 2023). 

*This repository is still in development*

## Installation

```
git clone 
cd f-PG
pip install -r requirements.txt
export PYTHONPATH=${PWD}:$PYTHONPATH
```

## Running

The repository contains code for running both the gridworld and the continuous domain experiments. The code for running gridworld is in ```grid_world_experiments```.

To run the baselines, 
```
python fpg/main_baselines.py --config-name <config>
```
and the run f-PG

```
python fpg/main_pg.py --config-name <config>
```

The ```configs``` contain the config files.

## Acknowledgements

The repository has used code from the [f-irl](https://github.com/iDurugkar/adversarial-intrinsic-motivation) and [AIM](https://github.com/iDurugkar/adversarial-intrinsic-motivation) repositories.

## Citation

If you like our work, please cite it as,
```
@inproceedings{agarwal2023fpg,
    author = {Agarwal, Siddhant and Durugkar, Ishan and Stone, Peter and Zhang, Amy},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {$f$ Policy Gradients: A General Framework for Goal Conditioned RL using $f$-Divergences},
    volume = {36}, 
    year = {2023}
}
```
