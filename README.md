# Low-Dose CT via Transfer Learning from a 2D Trained Network

This repository contains the code for CPCE-3D network introduced in the following paper

[3D Convolutional Encoder-Decoder Network for Low-Dose CT via Transfer Learning from a 2D Trained Network](https://doi.org/10.1109/TMI.2018.2832217) (IEEE TMI)

## Installation
Make sure you have [Python](https://www.python.org/) installed, then install [TensorFlow](https://www.tensorflow.org/install/) and [Scikit-learn](http://scikit-learn.org/) on your system.

## Usage

### Prepare the training data

In order to start the training process, please prepare your ``training data`` in the following form:

* ``data``: N x D x W x H
* ``label``: N x W x H 

Here N, D, W, and H are number, depth, width, and height of the input data, respectively. Each label corresponds to the central slice of input data. Then ``data`` and ``label`` are stored in a ``hdh5`` file.

### Pre-trained VGG model

Please also download the pre-trained VGG model from [here](https://mega.nz/1a61c789-50d3-42f8-92af-09f89e3a152e).

### Training network
```
python main.py
``` 

If you want to use the transfer learning from 2D to 3D, please train a 2D model first. The ``CPCE-3D`` model here can automatically deal with 2D input and 3D input with various depth (3, 5, 7, and 9), relying on the input size.

## Citation

If you found this code or our work useful please cite us

```
@article{shan20183d,
  title={3-D Convolutional Encoder-Decoder Network for Low-Dose CT via Transfer Learning from a 2-D Trained Network},
  author={Shan, Hongming and Zhang, Yi and Yang, Qingsong and Kruger, Uwe and Kalra, Mannudeep K and Sun, Ling and Cong, Wenxiang and Wang, Ge},
  journal={IEEE Transactions on Medical Imaging},
  volume={37},
  number={6},
  pages={1522--1534},
  year={2018},
  publisher={IEEE}
}
```

## Contact

shanh at rpi dot edu

Any discussions, suggestions and questions are welcome!



