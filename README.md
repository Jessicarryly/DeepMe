
# DeepMe

## Introduction
DeepMe is a neural network design for [2017 Physionet Challenge](https://physionet.org/challenge/2017/#preparing) on ECG classifcation

## Install
```shell
$ git clone https://github.com/dattran2346/DeepMe.git
$ cd DeepMe
$ ./setup.sh
```

## Usage
```shell
$ python deepme.py <option>
```

#### Options:
- train
- develop
- test
- path/to/.mat

## Demo

<img src=screenshot/demo.gif width=366 alt='Video Walkthrough' />


## Performance
v0.1: Simple 1 layer logistic regression

<img src=screenshot/v0.1.png width='256' alt='image' />

v0.2: [affine - relu -pool] - [affine - relu] - [fc] - softmax

<img src=screenshot/v0.2.png width='256' alt='image' />

v0.3: Preprocess data using Fast Fourier Transform

<img src=screenshot/v0.3.png width=256 alt=image />

## Inspiration

[DeepHeart physionet challeng 2016](https://github.com/jisaacso/DeepHeart)

[Tensorflow MNIST tutorial](https://www.tensorflow.org/get_started/mnist/pros)

## License
    Copyright [2017] [Ch4ul3n3]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
