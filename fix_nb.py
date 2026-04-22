import json

file_path = "model_predict_cavity_claw_RouteMeander_eigenmode/ml_22_print_results_surrogate_defined_loss.ipynb"
with open(file_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if "source" in cell:
        for i, line in enumerate(cell["source"]):
            line = line.replace("qubit-TransmonCross-Hamiltonian_params", "cavity_claw_RouteMeander_eigenmode")
            line = line.replace("Hamiltonian_column_names", "eigenmode_column_names")
            line = line.replace("Hamiltonian_names", "eigenmode_names")
            line = line.replace("Hamiltonian_name", "eigenmode_name")
            line = line.replace("n_Hamiltonian_params", "n_eigenmode_params")
            line = line.replace("Hamiltonian_pred", "eigenmode_pred")
            line = line.replace("Hamiltonian_abs_errors", "eigenmode_abs_errors")
            line = line.replace("Hamiltonian_sq_errors", "eigenmode_sq_errors")
            line = line.replace("Hamiltonian", "eigenmode")
            line = line.replace("hamiltonian", "eigenmode")
            
            # Fix scaler prefixes for the resonator model
            line = line.replace("x_scaler_prefix = 'scaler_X'", "x_scaler_prefix = 'scaler_X_linear'")
            line = line.replace("scaler_y_{col_name}_one_hot_encoding.save", "scaler_y_linear_{col_name}.save")
            
            # The transmon outputs are ['qubit_frequency_GHz', 'anharmonicity_MHz']
            # The resonator outputs are ['cavity_frequency', 'kappa']
            # The script above already replaces "Hamiltonian outputs..." string but there might be places printing it.
            
            cell["source"][i] = line

with open(file_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Done fixing notebook")
