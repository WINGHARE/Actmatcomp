# Active matcomp

This is a framework that combines categorical matrix completion and active machine learningto guide the high throughput screening experiment.

## Prerequisites

Here are some prerequisites to run the software
```
Python3.6
Numpy
Scipy
Pandas
Fancyimpute*
```

For the package:
* [Fancyimpute](https://github.com/iskandr/fancyimpute) 

Some base implmentation of the matrix completion related tasks. We integrate some codes in our projects and do some modifications.

## Installing

You may download the codes to the directory you want.
One thing to notice is that you may download the data for the experiment for real data in the homepage of the previous study:

* [Active Learning Of Perturbations](http://murphylab.web.cmu.edu/software/2016_eLife_Active_Learning_Of_Perturbations/) 

Download data following the guid and copy the following file to the 'data/calculated' directory:

```
zscored_data.npy
indices.npy
```


## Running the tests

You may run the codes on Python commond line.

```
python activelearner.py 
```
Options are displayed in the code.

## Authors

* **Junyi Chen** 

## License

Copyright <2018> <Junyi CHEN>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You may obtain a copy of the License at:

     https://opensource.org/licenses/BSD-3-Clause


## Group homepage

* [Group homepage](http://bioinfo.cs.cityu.edu.hk/)