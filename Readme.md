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

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Group homepage

* [Group homepage](http://bioinfo.cs.cityu.edu.hk/)