# tempEgo

tempEgo is a Python pacakge for calculating the ego-motion velocity of a sensing platform from millimetre-wave (mmWave) radar data. This package impliments the ego-motion estimation methods detailed in [Enhancing Doppler Ego-Motion Estimation: A Temporally Weighted Approach to RANSAC](link to be added).


The three methods are:
* KB: Traditional ego-motion estimation technique as detailed in D. Kellner et al., "[Instantaneous ego-motion estimation using doppler radar](https://doi.org/10.1109/ITSC.2013.6728341)," in 16th ITSC, 2013, pp. 869–874
* TEMPSAC: TEMporal SAmpling Consensus implements a weighted sliding window of sensor measurements biasing the algorithm to estimate the velocity using newer samples in time.
* TWLSQ: Temporally Weighted Least Squares implements a sliding window and temporal weighting scheme for model fitting.

## Prerequisites
A method of supplying this package with mmWave data, either using a physical sensor or a pre-existing dataset. 

For use with the Coloradar dataset please install the desired dataset from [here](https://arpg.github.io/coloradar/). For out of the box usage of this package with the Coloradar dataset follow the directory structure implimented [here](https://github.com/azinke/coloradar.git). If you do not need the extra functionality of the coloradar_package simply clone the repository to get the correct directory structure. 


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

The _main.py_ script provides an example of how to use the each velocity esimation method.

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

## License

[MIT](https://choosealicense.com/licenses/mit/)
