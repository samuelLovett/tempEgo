# tempEgo

tempEgo is a Python package for calculating the ego-motion velocity of a sensing platform from millimetre-wave (mmWave) radar data. This package implements the ego-motion estimation methods detailed in [Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC](link to be added).


The three methods are:
* KB: Traditional ego-motion estimation technique as detailed in D. Kellner et al., "[Instantaneous ego-motion estimation using doppler radar](https://doi.org/10.1109/ITSC.2013.6728341)," in 16th ITSC, 2013, pp. 869–874
* TEMPSAC: TEMporal SAmpling Consensus implements a weighted sliding window of sensor measurements biasing the algorithm to estimate the velocity using newer samples in time.
* TWLSQ: Temporally Weighted Least Squares implements a sliding window and temporal weighting scheme for model fitting.

## Prerequisites
A method of supplying this package with mmWave data, either using a physical sensor or a pre-existing dataset. 

For use with the Coloradar dataset please install the desired dataset from [here](https://arpg.github.io/coloradar/). For out of the box usage of this package with the Coloradar dataset follow the directory structure implemented [here](https://github.com/azinke/coloradar.git). If you do not need the extra functionality of the coloradar_package simply clone the repository to get the correct directory structure. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tempEgo.

```bash
pip install tempEgo
```
or

Clone the Github Repository

Prereq: [Git installation](https://github.com/git-guides/install-git)
```bash
git clone https://github.com/samuelLovett/tempEgo.git
cd tempEgo
pip install .
```



## Usage

The _main.py_ script provides an example of how to use each velocity estimation method.

**Hyperparameters:**

Shared:
* n: Minimum number of data points to estimate parameters
* k: Maximum iterations allowed
* epsilon: Threshold value to determine if points are fit well
* z: Number of close data points required to assert model fits well

TEMPSAC & TWLSQ:
* buffer_size: Size of the sliding window (m)
* fff_lambda: fixed forgetting factor as defined in (14) of [Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC](link to be added). 0 uses only the most recent samples and 1 uses all samples equally.

To change the from the values presented in [Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC](link to be added), change their definition within the _set_kb()_ , _set_tempsac()_ , _set_twlsq()_ methods.

## Results

Comparison of average absolute pose error for different environments. The average absolute pose error is presented in metres.

| Algorithm  | Edgar Classroom Dataset | Intelligent Robotic Lab Dataset  | Edgar Army tunnel Dataset |
| ------------- |:-------------:| ------------- |:-------------:|
| TEMPSAC      | 2.812     | 2.352      | **5.615**     |
| TWLSQ      | **2.338**     | **2.166**      | 5.653     |
| KB      | 3.644     | 3.603      | 5.672     |

The average absolute pose error for 100 trials is shown in the above Table. Both our methods outperform KB, with TEMPSAC showing an average improvement of 19.5% and TWLSQ showing an average improvement of 25.3%.
## License

[MIT](https://choosealicense.com/licenses/mit/)