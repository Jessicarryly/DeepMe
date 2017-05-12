
# DeepMe

## Introduction
DeepMe is a neural network design for [2017 Physionet Challenge](https://physionet.org/challenge/2017/#preparing) on ECG classifcation
 
## Install
```shell
$ git clone https://github.com/dattran2346/DeepMe.git
$ cd DeepMe
$ wget https://physionet.org/challenge/2017/training2017.zip
$ unzip training2017
$ pip install -r requirements.txt
$ python model.py
```

## Performance
v0.1: Simple 1 layer logistic regression

<img src=screenshot/v0.1.png width='256' alt='image' />

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
