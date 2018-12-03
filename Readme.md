# Active matcomp

This is a framework that combines categorical matrix completion and active machine learningto guide the high throughput screening experiment.

## Prerequisites

Here are some prerequisites to run the software
```
Python 3.6.2
Numpy
Scipy 
Pandas
Sklearn
Fancyimpute*
```

For the package:
* [Fancyimpute](https://github.com/iskandr/fancyimpute) 

Some base implmentation of the matrix completion related tasks. We integrate some codes in our projects and do some modifications.

## Installing

You may download the codes to the directory you want.
One thing to notice is that you may download the data for the experiment for real data in the homepage of the previous study:

* [Active Learning Of Perturbations](http://murphylab.web.cmu.edu/software/2016_eLife_Active_Learning_Of_Perturbations/) 

Download data from the file 'eLife_AL_intermediate_data.tar' following the guide and copy the following file to the 'data/calculated' directory:

```
zscored_data.npy
indices.npy
```


## Running the tests

You may run the codes on Python commond line.

```
python activelearner.py 
```
Options are displayed in the code. We have already set proper default setting. Do no modify the options unless you know the meaning of options.

## Authors

* **Junyi Chen** 

## License

To be determined


## Group homepage

* [Group homepage](http://bioinfo.cs.cityu.edu.hk/)