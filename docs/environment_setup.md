# Environment Setup

These instructions assume you are working from the repository root.

## Conda Environment

Install Conda or Miniconda first if it is not already available on your machine.

Create the environment with either:

```bash
conda env create -f environment.yml
```

or:

```bash
./scripts/environment/create_conda_env.sh
```

Activate it with:

```bash
conda activate cryo-modelling-env
```

This environment file now covers the core ML notebooks plus the checked-in paper figure scripts. In particular, it explicitly includes the packages used throughout the repo for:

- numerical work: `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`
- training/tuning: `tensorflow-cpu`, `tensorflow-datasets`, `keras-tuner`
- figures: `matplotlib`, `cairosvg`, `pymupdf`, `svglib`, `reportlab`

Remove it with either:

```bash
conda env remove --name cryo-modelling-env --all
```

or:

```bash
./scripts/environment/remove_conda_env.sh
```

## Fermilab EAF / Remote Jupyter

If you are running the notebooks on Fermilab EAF or another remote host:

1. Create and activate the conda environment.
2. Initialize Conda for your shell if needed:

```bash
conda init bash
source ~/.bashrc
```

3. Generate Jupyter configuration and set a password:

```bash
conda activate cryo-modelling-env
jupyter-lab --generate-config
jupyter-lab password
```

4. Start JupyterLab on an open port:

```bash
jupyter-lab --no-browser --ip 0.0.0.0 --port=8081
```

5. From your local machine, forward that port over SSH:

```bash
ssh -L 8888:localhost:8081 <REMOTE_USER>@<REMOTE_HOST>
```

Then open `http://localhost:8888` in your browser.

## GPU Note

This repo currently installs `tensorflow-cpu` from `environment.yml`, so GPU usage is not configured by default in the checked-in environment file. If you need GPU-backed TensorFlow on a remote system, treat that as an environment-specific override rather than the repository baseline.

## Validation Notebook Caveat

The validation notebooks are not fully covered by `environment.yml`. They also import the SQuADDS and Ansys stack, including packages such as `squadds` and `qiskit_metal`, and some of those workflows depend on external Ansys tooling as well.

That means:

- the updated environment file is enough to get started with the core training, evaluation, and plotting notebooks
- it is not, by itself, enough to run every validation notebook end to end

If you want the validation workflows to be reproducible from scratch, the repo still needs a dedicated, explicit install recipe for that external toolchain.
