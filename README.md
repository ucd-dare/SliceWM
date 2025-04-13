# World-Model-Based Adaptive Network Slicing

We introduce a world-model-based RL framework for adaptive network slicing, leveraging DreamerV3 based world model.

## Installation

### Install Conda

First, install Conda by following the instructions on the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html). You can skip this part if you have already installed Conda.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Then initialize your Miniconda on bash shells:

```bash
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Create a Conda Environment

Create a new Conda environment by running:

```bash
conda create -n slice python=3.8 -y
```

Activate the environment by running:

```bash
conda activate slice
```

Install the environment for WM-based network slicing.

```bash
pip install -r requirements.txt
```

## Training and Evaluation

The environment is located at the *embodied\envs\slice_env.py*. You can change the system dynamic and reward function by adjusting the parameter in the script.

To train the World Model, run the script
```bash
python train.py
```

To visualize the training reward, you can use tensorboard to monitor the training process:
```bash
tensorboard --logdir ./logdir
```

To evaluate the World Model, specify the key `from_checkpoint` in the configuration file `*dreamerv3/configs.yaml*` by the path of the checkpoint to be evaluated. For example:

```yaml
from_checkpoint: '/home/sliceWM/logdir/108-slice/checkpoint.ckpt'
```

Then run evaluation script by: 
```bash
python eval.py
```

The evaluation script records all the metrics in a csv file `recorded_metrics.csv`. To calculate the metrics in the evaluation, run the script:
```bash
python metric_cal.py
```
**Note:** If the csv file `recorded_metrics.csv` exists before running the evaluation script, the metrics from two evaluations mix. To avoid this, you can delete or rename the `recorded_metrics.csv` manually everytime before evaluation.

You can also further visualize the plot by:
```bash
python metric_plot.py
```