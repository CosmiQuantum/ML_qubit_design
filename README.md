# ML based Qubit Design

## Desired Flow

Below is the desired flow for this project. All three models will be stitched together to predict a Qiskit Metal design when given a set of desired "Top_Level_X" Hamiltonian values. This will be done using the "X_2.0" values that are simulated from the 'y' values predicted with each of the 3 individual models:

![Desired Flow](desired_flow.png)

## Environment Setup

Do this first. See the [ENV_SETUP](ENV_SETUP.md) documentation.

## Data Setup

Do this second :-) 

See the documentation in the [data](data) directory.

## Quick Start

If you have set up the environment and data, then move to the data analysis notebook (`00_`), then the model training (`01_`), etc.

To open your JupyterLab and then the notebooks:

```
conda activate qubit-design-env
jupyter-lab
```
## ToDo
- Update with reccomendations from Giuseppe to reduce randomness and introduce more determinicity into training
- Make third set of scripts that take the "X" values and map to the basic "Top_level_X" values
- Try to find other equations to map from "X" to "Top_level_X"
- Get Simulator pipeline set up to take predicted "y" values to "X_2.0" values

