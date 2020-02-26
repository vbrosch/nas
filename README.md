# NAS
A sample implementation for regularized evolution and progressive neural architecture search using PyTorch. Based on Google's [regularized evolution implementation](https://github.com/google-research/google-research/tree/master/evolution/regularized_evolution_algorithm) and
the paper of [Real et al. (2018)](https://arxiv.org/abs/1802.01548).

## Running regularized evolution

In order to run our regularized evolution algorithm, you can use the following command:

```
$ python3 regularized_evolution/main.py --cycles 100
```

The search can be configured by a variety of parameters.
Please note that it's necessary to install the requirements before:

```
$ pip install -r requirements.txt
```

## Running PNAS

Please note that PNAS is still work in progress, although it should be possible
to run it by calling:

```
$ python3 pnas/main.py
```
