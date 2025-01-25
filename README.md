# Deep Learning for Time Series Forecasting

## Creating Conda Enviorment

```
conda env create -f dlts.yml
```

Alternatively, you can create a new conda environment with Python 3.10 and installing (conda install...) the rest of packagues (see requirements.txt).

```
conda create --name name-of-env python=3.10
```

(Note: it is recommended to run the code in VS Code).

## DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

Paper: https://arxiv.org/abs/1704.04110.

Code from: https://github.com/husnejahan/DeepAR-pytorch.

### Instructions to run DeepAR

1. Download the dataset: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
2. Move it into the DeepAR folder and run:
```
python preprocess_elect.py
```

If training is taking to much, you can use just max_samples for training (but take into account that performance on test will drop):

```
sampler = WeightedSampler(data_dir, args.dataset, replacement=True, max_samples=500)
```
3. Run main.py: `python main.py` (if you run from terminal instead of VS Code you may have to change the directories used in the code)
4. Results will be saved in `DeepAR/experiments/base_model/`.

## Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting

Paper: https://arxiv.org/abs/1907.00235.

### Instructions to run ConvTras


1. Run main.py: `python main.py` (if you run from terminal instead of VS Code you may have to change the directories used in the code)

If training is taking to much, you can set the argument "train-ins-num" to a lower number, it will limit the number of time-series used for training (but take into account that performance on test will drop).


