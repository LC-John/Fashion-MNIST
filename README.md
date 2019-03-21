# Fashion MNIST

## The Fashion MNIST Dataset

The Fashion MNIST dataset is from [here](https://www.kaggle.com/zalando-research/fashionmnist).

It is a dataset a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Each training and test example is assigned to one of the following labels: t-shirt/top (0), trouser (1), pullover (2), dress (3), coat (4), sandal (5), shirt (6), sneaker (7), bag (8), ankle boot (9). The examples randomly sampled from the training set are listed as below.

<div style="text-align: center">
<img src="./images/fmnist_0.jpg"/>
<img src="./images/fmnist_1.jpg"/>
<img src="./images/fmnist_2.jpg"/>
<img src="./images/fmnist_3.jpg"/>
<img src="./images/fmnist_4.jpg"/>
<img src="./images/fmnist_5.jpg"/>
<img src="./images/fmnist_6.jpg"/>
<img src="./images/fmnist_7.jpg"/>
<img src="./images/fmnist_8.jpg"/>
<img src="./images/fmnist_9.jpg"/>
</div>

The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

## The CNN model

A just-so-so CNN model is provided. It applies the same structure as [this model](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-mnist-tutorial/mnist_4.2_batchnorm_convolutional.py).

Source code can be found in ``./code/model.py``, and the model checkpoint is in ``./model/model.ckpt``.

## Environments

If your package version is not the same as listed below, there is a chance that you may run the code successfully. But still, please use the recommended environment setting.

| Environment | Version |
| ----------- | ------- |
| python      | 3.6.6   |
| tensorflow  | 1.10.0  |
| numpy       | 1.16.0  |
| mnist       | --      |

The ``mnist`` package is a mnist data parser, which can be installed with ``pip install python-mnist``. It can be imported with ``import mnist`` in your own code.

## Project requirements

See [here](https://github.com/LC-John/Fashion-MNIST/blob/master/writeup/writeup.pdf).
