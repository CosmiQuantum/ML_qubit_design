# ML based Qubit Design

Contributors:
- Olivia Seidel (UT Arlington/Fermilab)
- Sara Sussman (Fermilab)
- Giuseppe Di Guglielmo (Fermilab)

## Features

This repo has tools/options to experiment with different configurations of a Multi-Layer Perceptron (MLP) for Qiskit Metal parameter prediction, when given a desired set of hamiltonian parameters. You can use the contained notebooks to explore the following options:

1. **Hyperparameter Optimization**  
   - Compare using **Keras Tuner** to automatically search for and return the best hyperparameters vs. using a predefined set.  

2. **Feature Scaling (0–1 normalization)**  
   - Option to scale all features to the range `[0, 1]` or leave them unscaled.  
   - I strongly recommended scaling because input values can vary drastically in magnitude for these Qiskit metal design parameters, and proper scaling often improves prediction accuracy across all parameters.  
   - Scaling amounts are saved, allowing you to safely reverse the scaling later and recover accurate/real-world predicted values.  

3. **Categorical Data Encoding**  
   - Compare **one-hot encoding** vs. **linear encoding** for categorical Qiskit Metal parameters.  
   - Since an MLP predicts numeric values rather than categories directly, categorical features must be converted into numeric form. These experiments help determine which encoding method gives better performance.

## Environment Setup

Do this first. See the [ENV_SETUP](ENV_SETUP.md) documentation.

## Quick Start

If you have set up the environment and data, then move to the data analysis notebook (`00_`), then the model training (`01_`), etc.

To open your JupyterLab and then the notebooks:

```
conda activate qubit-design-env
jupyter-lab
```

## Large Files

Some model checkpoints and intermediate datasets are too big for GitHub. You can grab everything you need from this Google Drive folder **[here](https://drive.google.com/drive/folders/1WVHR4b4g1M4xdOUghbwNKrABafRz-YaQ?usp=sharing)**.

The folder contains three directories—simply copy them into the root of your cloned repo. They’re already in `.gitignore`, so you won’t accidentally commit them, and you can dive straight into the notebooks without rerunning Keras Tuner or the data-processing steps.

If you have any access issues please send me a quick note at **olivias@fnal.gov**.

## Structure
The following three folders contain scripts to use Machine Learning to predict Qiskit design parameters based on target hamiltonian values for each specified design part:

- *model_predict_cavity_claw_RouteMeander_eigenmode*

- *model_predict_coupler_NCap_cap_matrix*

- *model_predict_qubit_TransmonCross_cap_matrix*

## Within each folder there are three scripts:

- *ml_00_data_analysis*:loads the data and parses it into a format to use in the next script

- *ml_01_train_keras*: Trains the model using an MLP

- *ml_hyperparameter_search_analysis*: provides plots to visualize the hyperparameter search, so the user can easily see the best values

## Reccomended optimization strategy:
In the *parameters.py* file start off with KERAS_TUNER=True, and after doing an automated hyperparameter search, grab the best hyperparameter values and update the *parameters.py* file. Then rerun with KERAS_TUNER=False to see how the model learns over epoch number.

## Completed
- Three models have been trained with optimized hyperparameters from keras-tuner, each model predicting Qiskit metal parameters for various parts of a transmon cross chip/resonator design
- Different encoding values were tested and optimized for the categorical y parameters in Qiskit Metal
- Different scaling techniques were implemented, where both X and Y are scaled
- Training and validation sets were explicitly seperated
  
## Desired Flow

Below is the desired flow for this project. All three models will be stitched together to predict a Qiskit Metal design when given a set of desired "Top_Level_X" Hamiltonian values. This will be done using the "X_2.0" values that are simulated from the 'y' values predicted with each of the 3 individual models:

![Desired Flow](desired_flow.png)
