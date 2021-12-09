# Covariance Steering Games with Squared Wasserstein Distance Cost 

Authors: Isin Balci & Martin Braquet.

This code has been designed during the project in Fall 2021 at the University of Texas at Austin in ASE 389: Modeling Multi-Agent Systems.

Please visit the website for more information: https://sites.google.com/utexas.edu/ut-ase389-stoch-games.

## Environment Setup

Create environment: `python3 -m venv env`

Activate environment: `source env/bin/activate`

Install requirements: `pip install -r requirements.txt`


## Simulation

### Iterative Linear Quadratic Gaussian

Run `python3 src/test_projectLQG.py` to obtain the simulation results for the LQG algorithm.

### Iterative Best Response

Run `python3 src/test_projectIBR.py` to obtain the simulation results for the IBR algorithm (for this, the MOSEK package requires a license which can be installed at https://www.mosek.com/products/academic-licenses/).

Then run `python3 src/plotIBR.py` to plot the results for the IBR algorithm.
