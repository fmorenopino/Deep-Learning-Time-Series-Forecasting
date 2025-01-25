# Deep Learning for Time Series Forecasting

This repository implements two deep learning models for time series forecasting: `DeepAR` ("DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks") and `ConvTrans` ("Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting"). See references below.

Both models follow a similar approach: first, they compute an embedding of the time series (DeepAR uses an LSTM, while ConvTrans employs a convolutional-Transformer). This embedded representation is then fed into an MLP, which outputs the forecast (mean and variance if modelled with a Gaussian distribution).

## Creating Conda Environment

The repo contains the `dlts.yml` file, a conda enviorment that allows to run both models. To install it:

```
conda env create -f dlts.yml
```

Alternatively, you can create a new conda environment with Python 3.10 (version used for testing) and installing (conda install...) the rest of packages (see requirements.txt).

```
conda create --name name-of-env python=3.10
```

(Note: it is recommended to run the code in VS Code).

## DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

- Paper: https://arxiv.org/abs/1704.04110.

- Code from: https://github.com/husnejahan/DeepAR-pytorch.

### Instructions to run DeepAR

1. Download the dataset: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
2. Move it into the DeepAR folder and run:
```
python preprocess_elect.py
```

If training is taking too much time, you can use the option `max_samples` when setting up the `WeightedSampler` for training on a subset of time series (but take into account that you probably won't converge to a proper solution and performance on test will drop):

```
sampler = WeightedSampler(data_dir, args.dataset, replacement=True, max_samples=5000)
```
3. Run main.py: `python main.py` (if you run from terminal instead of VS Code you may have to change the directories used in the code).
4. Results will be saved in `DeepAR/experiments/base_model/`.



## Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting

- Paper: https://arxiv.org/abs/1907.00235.

### Instructions to run ConvTras


1. Run main.py: `python main.py` (if you run from terminal instead of VS Code you may have to change the directories used in the code)

If training is taking too much time, you can set the argument "train-ins-num" to a lower number, it will limit the number of time-series used for training (but take into account that performance on test will drop).



