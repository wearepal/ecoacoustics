# Deep Learning Soundscapes for Ecoacoustics

This repository packages up the work I undertook for my internship with the Predictive Analytics Lab at Sussex University. The repository demonstrates how to use the Predictive Analytics Lab's Conduit framework integration of the ecoacoustics audio dataset provided by Eldrige et al. [1], while also reproducing some of the results obtained by Sethi et al. [2].

## Installation
First ensure you have a Python environment (3.8) either via Anaconda
(instructions https://docs.anaconda.com/anaconda/install/) or pre-installed (e.g. on Linux).
Dependencies for this repository are managed with [Poetry] and it is recommended to install dependencies in a virtual environment as Conduit has quite a few dependencies. The following steps will show how this is done using [Anaconda].


Instructions for Anaconda:

1.  Create and activate an Anaconda virtual environment (`python=3.8` installs Python version 3.8):
```sh
    conda update conda
    conda create --name my_env python=3.8
    conda activate my_env
    pip install --upgrade pip
```
2. Install poetry by running the approporiate command for your system:
    * osx / linux / bashonwindows install instructions:
        ```sh
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
        ```
    * windows powershell install instructions:
        ```sh
        (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
        ```
    * (Check for updates) `poetry update self`


Instructions for pre-installed Python (3.8):

```
# Create your environment
python -m venv my_env
# Activate it
source my_env/bin/activate
# Upgrade pip (important to get the right versions!)
pip install --upgrade pip
# Install poetry
pip install poetry
```




1. Once you've got an environment set up, and poetry installed, you can clone this repository to your machine,
and change directory to base directory of the repo (ensure you are in the same directory as the pyproject.toml file):

```
    git clone https://github.com/predictive-analytics-lab/ecoacoustics.git
    cd ecoacoustics
```
2. Call poetry to install all modules/packages that this repository depends on:
    * As a user: `poetry install --no-dev`
    * As a developer: `poetry install`
3. Either run the cells in the notebook manually or run the script with:
    ```sh
    python '.\ecoacoustics\process_vggish.py'
    ```
   ```shell
    python main.py
    ```
---
[1]: Alice Eldridge, Paola Moscoso, Patrice Guyot, & Mika Peck. (2018). Data for "Sounding out Ecoacoustic Metrics: Avian species richness is predicted by acoustic indices in temperate but not tropical habitats" (Final) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1255218

[2]: Sethi, Sarab S., Nick S. Jones, Ben D. Fulcher, Lorenzo Picinali, Dena Jane Clink, Holger Klinck, C. David L. Orme, Peter H. Wrege, and Robert M. Ewers. “Characterizing Soundscapes across Diverse Ecosystems Using a Universal Acoustic Feature Set.” Proceedings of the National Academy of Sciences 117, no. 29 (July 21, 2020): 17049–55. https://doi.org/10.1073/pnas.2004702117.

[//]: #
  [Poetry]: <https://python-poetry.org/>
  [Anaconda]: <https://docs.anaconda.com/anaconda/>
