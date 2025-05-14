# ML based Qubit Design

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

## Large Files
To back up models that are to large for github, I use the drive [here](https://drive.google.com/drive/folders/1WVHR4b4g1M4xdOUghbwNKrABafRz-YaQ?usp=sharing). Please email olivias@fnal.gov if you can't access it for some reason.

## Structure
The following three folders contain scripts to use Machine Learning to predict Qiskit design parameters based on target hamiltonian values for each specified design part:
*model_predict_cavity_claw_RouteMeander_eigenmode*
*model_predict_coupler_NCap_cap_matrix*
*model_predict_qubit_TransmonCross_cap_matrix*

Within each folder there are three scripts:
*ml_00_data_analysis*:loads the data and parses it into a format to use in the next script
*ml_01_train_keras*: Trains the model using an MLP
*ml_hyperparameter_search_analysis*: provides plots to visualize the hyperparameter search, so the user can easily see the best values

## Completed
- Three models have been trained with optimized hyperparameters from keras-tuner, each model predicting Qiskit metal parameters for various parts of a transmon cross chip/resonator design
- Different encoding values were tested and optimized for the categorical y parameters in Qiskit Metal
- Different scaling techniques were implemented, where both X and Y are scaled
- Training and validation sets were explicitly seperated
  
## Desired Flow

Below is the desired flow for this project. All three models will be stitched together to predict a Qiskit Metal design when given a set of desired "Top_Level_X" Hamiltonian values. This will be done using the "X_2.0" values that are simulated from the 'y' values predicted with each of the 3 individual models:

![Desired Flow](desired_flow.png)