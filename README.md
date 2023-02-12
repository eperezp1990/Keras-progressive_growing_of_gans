# What you can expect from this repository

This repository has been used as the baseline implementation for the manuscript entitled "Progressive growing of Generative Adversarial Networks for improving data augmentation and skin cancer diagnosis".


## Requirements

1. Python3 
2. keras 2.1.2 (TensorFlow backend)


## How to run (For other Users)

### 1. Clone the repository

### 2. Prepare the dataset

Run **h5tool.py** to create HDF5 format datatset. Default settings of **h5tool.py** will crop the picture to 512*512, and create a channel-last h5 file.

Modify **config.py** to your own settings.

```
# In config.py:
data_dir = 'datasets'
result_dir = 'results'

dataset = dict(h5_path=<h5 file name>, resolution=128, max_labels=0, mirror_augment=True)
# Note: "data_dir" should be set to the direcory of your h5 file.
```


### 3. Begin training
```
$ python3 train.py
```

In **train.py**:

```
# In train.py:
speed_factor = 20
# set it to 1 if you don't need it.
```

"speed_factor" parameter will speed up the transition procedure of progressive growing of gans(switch resolution), at the price of reducing images' vividness, this parameter is aimed for speed up the validation progress in our development, however it is useful to see the progressive growing procedure more quickly, set it to "1" if you don't need it.

### 4.Save and resume training weights

The operations are the same as VStdio/VSCode users.

**So far, if your settings have no problem, you should see running information like our [running_log_example](Example/running_log_example.txt)**


## License

Our code is under [MIT license](https://en.wikipedia.org/wiki/MIT_License). See [LICENSE](LICENSE)
