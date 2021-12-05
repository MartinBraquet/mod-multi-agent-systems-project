# Covariance Steering Games with Squared Wasserstein Distance Cost 

Authors: Isin Balci & Martin Braquet.

This code has been designed during the project in Fall 2021 at the University of Texas at Austin in ASE 389: Modeling Multi-Agent Systems.

Please visit the website for more information: https://sites.google.com/utexas.edu/ut-ase389-stoch-games.

## Environment Setup

Create environment: `python3 -m venv env`

Activate environment: `source env/bin/activate`

Install requirements: `pip install -r requirements.txt`

## Simulation

Run `test_projectLQG.py` to obtain the simulation results for the LQG algorithm.

Run `test_projectIBR.py` to obtain the simulation results for the IBR algorithm.

Run `LQFeedbackFunction(A, B1, B2, Q1, Q2, R1, R2, x1, horizon)` to obtain the feedback LQ solution of a two-agent system.
