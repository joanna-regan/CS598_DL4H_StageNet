# CS598 Deep Learning For Healthcare - Reproducibility Project

This repository contains the code used to reproduce the main model, as well as ablations and statistics, for the proposed StageNet model. The purpose of this repository is to act as supplementary material to satisfy requirements for the Final Project in the Spring 2023 offering of CS598 - Deep Learning for Healthcare, taught by Professor Jimeng Sun at the University of Illinois - Urbana-Champaign.

## Reference to original sources

We reproduce results conducted by the original authors of [StageNet: Stage-Aware Neural Networks for Health Risk Prediction](https://arxiv.org/pdf/2001.10054.pdf).

In this reproducibility study, we leverage the author's original code where available. Their repository can be found at: https://github.com/v1xerunt/StageNet.

Please be sure to reference the original paper:
```
Junyi Gao, Cao Xiao, Yasha Wang, Wen Tang, Lucas M. Glass, Jimeng Sun. 2020. 
StageNet: Stage-Aware Neural Networks for Health Risk Prediction. 
In Proceedings of The Web Conference 2020 (WWW ’20), April 20–24, 2020, Taipei, Taiwan. ACM, New York, NY, USA, 11 pages. 
https://doi.org/10.1145/3366423.3380136
```

## Dependencies

Per the original author's instruction, we recommend:
* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA

## Data Download Instructions

We use the MIMIC-III Clinical Database for this study, downloaded from https://physionet.org/content/mimiciii/1.4/. The dataset is free and publicly available, though users must first be credentialed by PhysioNet and sign the Data Usage Agreement.

We download and unzip all CSV files. More download options can be found under the 'Files' section on the PhysioNet website.

## Data Preprocessing instructions

After download, we must build the benchmark dataset for the decompensation prediction task. We follow the directions as set forth by https://github.com/YerevaNN/mimic3-benchmarks/. This repository was copied into the ```mimic4extract``` folder of the current repository.

For reference, we give the approximate times to build the benchmark on our local machine:
* Downloading data: few minutes
* Step 1: generating stays, diagnoses, and events: 2 hours
* Step 2: fix missing data/events: 2.5 hours
* Step 3: generating episode # and episode # timeseries csv files: 5.5 hours
* Step 4: split into train and test sets: few minutes
* Step 5: generate decompensation-specific dataset: 1 hour
* Step 6: split training set into train and validation: few seconds

Once the benchmark sets are build, we store the training data in ```./data/train```, the test data in ```./data/test```, and we store the 3 ```listfile.csv``` files (for train, test, and validation) in ```./data```.

We then create a new folder ```./data/train_subdivided``` and run ```subdivide_train_data.py```. This will subdivide the training data set into multiple sub-folders and put them in ```./data/train_subdivided```. This step was added to help ease computational restraints when working with the data in Google Colab.

We provide sample data in the folders to help users understand the expected data structures.

## Training & Evaluation Code

We've created 3 similar scripts to train the original StageNet model and to train 2 ablation models. In order to run these, we must provide the dataset directory and file name to save the best model from each. 

For the purposes of reproducibility, we further specify the number of epochs and the subsampling size (i.e. how large we want our training data set to be).

```$ python train.py --data_path='./data/' --file_name='trained_model' --epochs=50 --small_part=5000 ```

```$ python train_ablation1.py --data_path='./data/' --file_name='trained_model_ablation1' --epochs=50 --small_part=5000 ```

```$ python train_ablation2.py --data_path='./data/' --file_name='trained_model_ablation2' --epochs=50 --small_part=5000 ```

Additional hyperparameters can also be specified such as ```--batch_size <integer> ```, learning rate ```--lr <float> ```, dimension of RNN ```--rnn_dim <integer> ```, convolution kernel size ```K <integer> ``` and more. For more information, try:

```$ python train.py --help```

For each of the training scripts, evaluation results on test data will be output after training completes.

## Pretrained models and Fast Evaluation

Pretrained models can be found in this repository under ```./saved_weights```. This subdirectory contains:
* ```StageNet```: pre-trained StageNet model provided by the original authors
* ```trained_model```: best pre-trained StageNet model created for the current reproducibility study (trained on subsample of data)
* ```trained_model_ablation1```: best pre-trained StageNet-I model created as an ablation for the current reproducibility study (trained on subsample of data)
* ```trained_model_ablation2```: best pre-trained StageNet-II model created as an ablation for the current reproducibility study (trained on subsample of data)

To evaluate these models with the test data, we can run the respective training scripts with ```test_mode=1``` as following:

For the original StageNet:
```$ python train.py --test_mode=1 --data_path='./data/' --file_name='StageNet'```

For the reproduced StageNet:
```$ python train.py --test_mode=1 --data_path='./data/' ```

For the first ablation model StageNet-I:
```$ python train_ablation1.py --test_mode=1 --data_path='./data/' ```

For the second ablation model StageNet-II:
```$ python train_ablation2.py --test_mode=1 --data_path='./data/' ```

## Results



