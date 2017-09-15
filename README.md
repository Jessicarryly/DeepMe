
# DeepMe

## Introduction
DeepMe is a neural network design for [2017 Physionet Challenge](https://physionet.org/challenge/2017/#preparing) on ECG classifcation. The data has ~ 8500 trainning set and ~ 300 validation set. All ECG recoding were sampled at 300 Hz and has been band pass filter (0.5 Hz - 50 Hz). I further decided to lowpass filter the signal with 40 Hz cutoff.

- Signal from physionet

<img src=screenshot/before.png width=512 alt='image' />

- Signal after my 40 Hz filter

<img src=screenshot/after.png width=512 alt='image' />
 
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
- test
- path/to/.mat

## Demo

<img src=screenshot/demo.gif width=512 alt='Video Walkthrough' />


## Performance
v0.1: Simple 1 layer logistic regression 

<img src=screenshot/v0.1.png width='512' alt='image' />

v0.2: [affine - relu -pool] - [affine - relu] - [fc] - softmax

<img src=screenshot/v0.2.png width='512' alt='image' />

v0.3: Add lowpass FIR filter with cutoff frequency of 40 Hz, default input size is 2^11 = 2048 to boost computation efficiency
<img src=screenshot/v0.3.png width='512' alt='image' />

When running with on GeForce GT 750M

<img src=screenshot/gpu.png width='256' alt='image' />

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

Linear Regression
Total train time 8.45653510094
Accuracy in the validation set is 0.260074066022
Testing time 13.9749529362

SimpleCNN
Total train time 3017.16695285
Accuracy in the validation set is 0.935226413798
Testing time 28.5029058456

AlexNet