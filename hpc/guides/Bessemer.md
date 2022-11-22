# Running code in Bessemer

## Connecting to Bessemer

First, connect to the HPC Bessemer cluster:

```sh
ssh -X <username>@bessemer.shef.ac.uk
```

where `<username>` is your MUSE username. Enter your MUSE password when prompted with `Password:` and authenticate through MFA.

## Setup

You'll need to clone this repository first:

```sh
git clone https://github.com/Haotian-Qi/conditional-probing-fork.git
```

Next, you need to enter an interactive job session in order to set up the Python environment. Do this by running the command:

```sh
srun --mem=8G --partition=gpu --qos=gpu --nodes=1 --gpus-per-node=1 --pty bash -i
```

This indicates that we are requesting 8GB of memory and one GPU for the session. This will allow the right amount of memory to be able to install PyTorch, and test our installation as well.

It might take a moment to load while resources are allocated for the session. You'll see the command prompt appear again once this has happened.

Once this happens, `cd` into the repository:

```sh
cd conditional-probing-fork
```

We need to set up your Python environment with the packages needed, and copy the datasets in the `/data` directory to allow faster access. Use the script `hpc/setup/bessemer.sh` to do this:

```sh
./hpc/setup/bessemer.sh
```

This will take a while as the packages are downloaded. Once that has finished, you can test that `torch` is working and that it has access to CUDA. Type `python3` in the command line and test for CUDA using:

```python
>>> import torch
>>> torch.cuda.is_available()
True
```

It should say `True` because we requested GPU for this interactive session.

## Submitting a job

Before submitting jobs, you'll need to enter an interactive job session using `srun --pty bash -i`, and `cd` into the repository. Test to see if you've got access to Python using `python3 --version`. If that command does not work, then you also need to activate the right environment module in order to run Python:

```sh
module load Anaconda3/5.3.0
```

From there, use the Python script `hpc/submit_job.py` to submit a job. The script takes two arguments: `config` is the path to the config `.yaml` file to use for the experiment, and `email` is the email address where updates about the submitted job should be sent. An example usage of this command would be:

```sh
python3 hpc/submit_job.py distilbert/configs/sst/bert-sst-layer0-0.yaml test@sheffield.ac.uk
```

It should show the path to the script file produced for running the submission, and the job ID for the submitted job.

Once the job has been submitted, you can submit more jobs, or you can quit the interactive job session by pressing `Ctrl+D`.

## Checking the status of a job

You can check the status of jobs that you have submitted using the command `sstat --user=$USER`, which shows you a table of the active jobs, including the status of them in the `ST` column.

- `PD` indicates that the job is being held in a queue awaiting resource allocation.
- `R` indicates that the job is running.
- `CD` indicates that the job has been completed.

You can find out more [here](https://docs.hpc.shef.ac.uk/en/latest/hpc/scheduler/index.html#id3).

## Checking the logs

ShARC creates logs in the directory where you submitted a job, i.e. the Git repository. You can check the output logs any submitted job when it has finished by looking in files named in the format:

```sh
${JOB_SCRIPT}-${JOB_ID}.out
```

So if the submission job script is called `job.sh` and the job ID is `123456`, then the logs would be called

```sh
job.sh-123456
```

## Further Reading

[HPC Documentation](https://docs.hpc.shef.ac.uk/en/latest/hpc/index.html)
