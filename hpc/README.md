# Submitting an HPC job

## Connecting to HPC

First, Connect using SSH:

```sh
ssh -X <username>@<cluster>.shef.ac.uk
```

where `<username>` is your MUSE username, and `<cluster>` is either `sharc` or `bessemer`. Enter your MUSE password when prompted, and follow the instructions to authenticate through Duo.

## Setup

You'll need to clone this repository:

```sh
git clone https://github.com/Haotian-Qi/conditional-probing-fork.git
```

Next, you'll need to need to enter an interactive session to setup the Python environment.

If you're on ShARC run the command:

```sh
qrshx -l rmem=8G -l gpu=1
```

If you're on Bessemer run the command:

```sh
srun --mem=8G --partition=gpu --qos=gpu --nodes=1 --gpus-per-node=1 --pty bash -i
```

This indicates that we are requesting 8GB of memory and one GPU for the session. This will allow the right amount of memory to be able to install PyTorch, and test our installation as well.

It might take a moment to load while resources are allocated for the session. You'll see the command prompt appear again once this has happened. Once this happens, `cd` into the repository:

```sh
cd conditional-probing-fork
```

We need to setup our Python environment with the packages needed. A script is provided for each cluster to do this. Run the command:

```sh
./hpc/setup/<cluster>.sh
```

replacing `<cluster>` with either `sharc` or `bessmer`.

This will take a while as the packages are downloaded. Once that has finished, you can test that PyTorch is working and that it has access to CUDA.

Do `source activate darwin` to activate the Conda environment where packages have been installed, then type `python3` in the command line and test for CUDA using:

```python
>>> import torch
>>> torch.cuda.is_available()
True
```

It should say `True` because we requested GPU for this interactive session.

## Submitting a job

Before submitting jobs, you'll need to enter an interactive job session.

Use `qrshx` if you're on ShARC, or `srun --pty bash -i` if you're on Bessemer.

Test to see if you've got access to Python using `python3 --version`. If that command does not work, then you also need to activate the right environment module in order to run Python.

On ShARC the command to do this is:

```sh
module load apps/python/conda
```

On Bessemer the command is

```sh
module load Anaconda3/5.3.0
```

Create a new plain tex file under the `hpc/` directory called `email` with your email address, e.g.

```plain
test@sheffield.ac.uk
```

From there, use the Python script `hpc/submit_job.py` to submit a job. The script takes a single positional arguemnt, `config`, which is the path to the config `.yaml` file to use for the experiment. The Python script automatically detects which cluster you're on.

An example usage of this command would be:

```sh
python3 hpc/submit_job.py distilbert/configs/sst/bert-sst-layer0-0.yaml
```

It should show the path to the script file produced for running the submission, and the job ID for the submitted job.

Once the job has been submitted, you can submit more jobs, or you can quit the interactive job session by pressing `Ctrl+D`.

## Checking the Status of a Job

You can check the status of jobs that you have submitted. The command varies between ShARC and Bessemer.

### ShARC

You can check the status of jobs you've submitted using the command

```
qstat
```

which shows you a table of the active jobs, including the status of them in the `state` column. `hw` indicates that the job is being held in a queue, while `r` indicates that the job is running. If you don't see a job in the table, then it means that the job has finished. 

You can find out more [here](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#monitoring-running-jobs).

### Bessemer

You can check the status of jobs you've submitted using the command

```sh
sstat --user=$USER
```

which shows you a table of the active jobs, including the status of them in the `ST` column.

- `PD` indicates that the job is being held in a queue awaiting resource allocation.
- `R` indicates that the job is running.
- `CD` indicates that the job has been completed.

You can find out more [here](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#id3).

## Checking the logs

The schedulers in both clusters create logs. Both create logs in the directory in which you submit the job.

### ShARC

ShARC creates two logs for each job. One for standard output, and one for standard error. You can check the output and error logs any submitted job when it has finished by looking in files named in the format:

```sh
${JOB_SCRIPT}.o${JOB_ID}
${JOB_SCIPRT}.e${JOB_ID}
```

So if the submission job script is called `job.sh` and the job ID is `123456`, then the logs would be called

```sh
job.sh.o123456
job.sh.e123456
```

### Bessemer

Unlike in ShARC, both standard output and error streams go into the same file. You can check the output logs any submitted job when it has finished by looking in files named in the format:

```sh
${JOB_ID}.out
```

So if the job ID is `123456`, then the logs would be called

```sh
123456.out
```

## Further Reading

[HPC Documentation](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html)
