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
- training/tuning: `tensorflow`, `tensorflow-datasets`, `keras-tuner`
- figures: `matplotlib`, `seaborn`, `cairosvg`, `pymupdf`, `svglib`, `reportlab`

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

1. Create and activate either the default conda environment or the EAF GPU environment below.
2. Initialize Conda for your shell if needed:

```bash
conda init bash
source ~/.bashrc
```

3. Generate Jupyter configuration and set a password:

```bash
conda activate cryo-modelling-env  # or cryo-modelling-env-gpu
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

## Fermilab EAF GPU Environment

The default `environment.yml` installs TensorFlow without the bundled NVIDIA CUDA pip packages. For Fermilab EAF GPU runs, create the GPU-specific environment instead:

```bash
conda env create -f environment-eaf-gpu.yml
conda activate cryo-modelling-env-gpu
```

That file uses `tensorflow[and-cuda]==2.20.0`, which is the GPU-capable TensorFlow install path used for the runtime benchmark notebooks on EAF.

The checked-in `ml_30_runtime_batch_benchmark` metadata records this successful EAF run:

- Python: `3.10.13`
- TensorFlow: `2.20.0`
- GPU visible to TensorFlow: `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`
- GPU hardware in the training notebook output: NVIDIA A100 80GB PCIe MIG 4g.40gb

The older metadata did not record the standalone Keras package version. New benchmark runs record it. You can also check it in any active environment with:

```bash
python -c "import platform, tensorflow as tf, keras; print('python', platform.python_version()); print('tensorflow', tf.__version__); print('keras', keras.__version__); print('gpus', tf.config.list_physical_devices('GPU'))"
```

## Validation Notebook Caveat

The validation notebooks are not fully covered by `environment.yml`. They also import the SQuADDS and Ansys stack, including packages such as `squadds` and `qiskit_metal`, and some of those workflows depend on external Ansys tooling as well.

That means:

- the updated environment file is enough to get started with the core training, evaluation, and plotting notebooks
- it is not, by itself, enough to run every validation notebook end to end

If you want the validation workflows to be reproducible from scratch, the repo still needs a dedicated, explicit install recipe for that external toolchain.
