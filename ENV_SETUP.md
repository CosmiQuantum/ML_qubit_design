# Environment setup
(From Giuseppe DiGuglielmo)

## Python

Install Conda on your host. You can download the most recent release of Miniconda3:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
For more details and a different release, visit Miniconda [download](https://docs.conda.io/en/latest/miniconda.html) and [installation](https://conda.io/projects/conda/en/stable/user-guide/install/index.html) pages.

Once Conda is installed, create the Python environment with the provided [configuration file](environment.yml):
```
conda env create -f environment.yml
```

or simply:
```
./create_env.sh
```
On Fermi EAF, youll then need to run:
```
conda init bash
source ~/.bashrc  
```

Activate it:
```
conda activate cryo-modelling-env
```
For more details about managing Conda environments see the [manual](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

To remove the environment:
```
conda remove --name cryo-modelling-env --all
```

or simply:
```
./remove_env.sh
```

## GPU Setup

The GPU setup may require some extra effort and the following instructions may not cover all of the corner cases.

If you have installed conda and create your environment `cryo-modelling-env`, _likely_ TensorFlow and GPU libraries and drivers are correctly installed. You can test it, first run 
```
conda activate cryo-modelling-env # if you don't have the environment active
ipython
```
then, in the `ipython` console:
```
import tensorflow as tf
tf.test.is_gpu_available()
```

If you get a message as
```
[...]
Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
[...]
```
You may try something like:
```
cd $HOME/miniconda3/envs/cryo-modelling-env/lib
ln libcusolver.so.11 libcusolver.so.10
cd -
```
Then try again the GPU test.

## Jupyter

If you are running these notebooks on a remote server without a display available:

```
conda activate cryo-modelling-env # if you don't have the environment active
jupyter-lab --generate-config
jupyter-lab password
```

At this point, you can run (please try to use a **DIFFERENT PORT** than your colleagues):
```
# jupyter-lab --no-browser --ip 0.0.0.0 --port=<PORT>
jupyter-lab --no-browser --ip 0.0.0.0 --port=8081
```

You can access the notebook from your remote machine over SSH by setting up a SSH tunnel. Run the following command from your _local machine_:

```
# Replace <PORT> with the port number you selected in the above step
# Replace <REMOTE_USER> with the remote server username
# Replace <REMOTE_HOST> with your remote server address
#ssh -L 8888:localhost:<PORT> <REMOTE_USER>@<REMOTE_HOST>
ssh -L 8888:localhost:8081 <REMOTE_USER>@<REMOTE_HOST>
```

Then point your browser to `https://localhost:<PORT>`. Once again, point to the `PORT` you chose when running the previous command.
