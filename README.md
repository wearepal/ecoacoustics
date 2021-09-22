# Deep Learning Soundscapes for Ecoacoustics

This repository packages up the work I undertook for my internship with the Predictive Analytics Lab at Sussex University. The repository demonstrates how to use the Predictive Analytics Lab's Conduit framework integreation of the ecoacoustics audio dataset provided by Aldrige et al. [1], while also reproducing some of the results obtained by Sethi et al. [2].

## Installation

Dependencies for this repository are managed with [Poetry] and it is recommended to install dependencies in a virtual environment as Conduit has quite a few dependencies. The following steps will show how this is done using [Anaconda].

1. Install poetry by running the approporiate command for your system:
    * osx / linux / bashonwindows install instructions:
        ```sh
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
        ```
    * windows powershell install instructions:
        ```sh
        (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
        ```
2. Install Anaconda by following their installation instructions:
    * https://docs.anaconda.com/anaconda/install/
3. Create and activate an Anaconda virtual environment (`python=3.9` isntalls Python version 3.9):
    ```sh
    conda create --name my_env python=3.9
    conda activate my_env
    ```
4. Clone this repository to your machine and change directory to base directory of the repo (ensure you are in the same directory as the pyproject.toml file):
    ```
    git clone https://github.com/predictive-analytics-lab/ecoacoustics.git
    cd ecoacoustics
    ```
5. Call poetry to install all modules/packages that this repository depends on:
    ```sh
    poetry install
    ```
6. Either run the cells in the notebook manually or run the script with:
    ```sh
    python '.\ecoacoustics\Ecoacoustics Script.py'
    ```
---
[1]: Alice Eldridge, Paola Moscoso, Patrice Guyot, & Mika Peck. (2018). Data for "Sounding out Ecoacoustic Metrics: Avian species richness is predicted by acoustic indices in temperate but not tropical habitats" (Final) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1255218

[2]: Sethi, Sarab S., Nick S. Jones, Ben D. Fulcher, Lorenzo Picinali, Dena Jane Clink, Holger Klinck, C. David L. Orme, Peter H. Wrege, and Robert M. Ewers. “Characterizing Soundscapes across Diverse Ecosystems Using a Universal Acoustic Feature Set.” Proceedings of the National Academy of Sciences 117, no. 29 (July 21, 2020): 17049–55. https://doi.org/10.1073/pnas.2004702117.

[//]: #
  [Poetry]: <https://python-poetry.org/>
  [Anaconda]: <https://docs.anaconda.com/anaconda/>
