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
  - Dataset 1 (w_resonator and kappa): These themselves are top level x values.
  - Dataset 2 (qubit-claw capacitance matrix): the cross-cross self-capacitance of the cross, C, maps to qubit anharmonicity by e^2/2C = E_c = - anharmonicity
  - Dataset 3 (coupler-feedline capacitance matrix): To do, later. Will be utilized in mapping back to "top level kappa"
- Make scripts that take the "Y" values and analytically check them against the basic "Top_level_X" values
  - Dataset 1 (we query for f_resonator and kappa total = kappa internal + kappa external):
    - Youtube tutorial of computing kappa internal, f_resonator from a resonator *layout*. that also links to paper and python calculator (https://www.youtube.com/watch?v=eQssX1e0TSU). The sanity check we want: f_resonator close to target, and the kappa returned by the youtube tutorial (kappa_internal) should be less than kappa we originally queried for 
  - Dataset 2 (qubit-claw capacitance matrix): N/A, no such resource already exists online that Sara knows of for the Y values here
  - Dataset 3 (coupler-feedline capacitance matrix): N/A, no such resource already exists online that Sara knows of for the Y values here
- Try to find other equations to map from "X" to "Top_level_X"
  - Sara will help with this. First we need to find for the SQUADDs paper device, "top level X" rough estimate, X and Y so that we can test these equations with real examples (the 6 qubits)
- Get Simulator (HFSS) pipeline set up to take predicted "y" values to "X_2.0" values
  - Sara
