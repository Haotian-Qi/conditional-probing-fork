#!/usr/bin/env python3

import os
import signal
import sys

import click
import torch
import yaml as yaml

from cache import *
from dataset import *
from model import *
from probe import *
from reporter import *
from task import *
from trainer import *
from utils import *
from utils import DEV_STR, TEST_STR, TRAIN_STR


class GracefulKill:
    def __init__(self, callback):
        self.callback = callback
        self.kill_handled = False

    def int_handler(self, signum, frame):
        self.kill_handled = True
        self.callback()
        signal.default_int_handler(signum, frame)

    def term_handler(self, signum, frame):
        self.kill_handled = True
        self.callback()
        sys.exit(1)

    def __enter__(self):
        signal.signal(signal.SIGINT, self.int_handler)
        signal.signal(signal.SIGTERM, self.term_handler)
        return self

    def __exit__(self, *args):
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, signal.SIG_DFL)


@click.command()
@click.argument("yaml_path")
@click.option(
    "--cache-data-only",
    is_flag=True,
    help="Writes data to cache and exits",
)
@click.option(
    "--do-test",
    is_flag=True,
    help="Evaluates on the test set as well as train and development sets",
)
def run_yaml_experiment(yaml_path, cache_data_only, do_test):
    """
    Runs an experiment as configured by a yaml config file
    """

    # Take constructed classes from yaml
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
    list_dataset = yaml_args["dataset"]
    list_model = yaml_args["model"]
    probe_model = yaml_args["probe"]
    regimen_model = yaml_args["regimen"]
    reporter_model = yaml_args["reporter"]

    # Make results directory
    os.makedirs(regimen_model.reporting_root, exist_ok=True)

    # Make dataloaders and load data
    list_dataset.before_load()

    with GracefulKill(list_dataset.after_load) as g:
        try:
            train_dataloader = list_dataset.get_train_dataloader(shuffle=True)
            dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
            if do_test:
                test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
        finally:
            if not g.kill_handled:
                list_dataset.after_load()

    if cache_data_only:
        print("Only caching datasets, exiting...")
        return

    # Train probe
    regimen_model.train_until_convergence(
        probe_model,
        list_model,
        None,
        train_dataloader,
        dev_dataloader,
        gradient_steps_between_eval=min(1000, len(train_dataloader)),
    )

    # Train probe with finetuning
    # regimen_model.train_until_convergence(probe_model, list_model
    #    , None, train_dataloader, dev_dataloader
    #    , gradient_steps_between_eval=1000, finetune=True)

    # Load best probe from disk
    probe_model.load_state_dict(torch.load(regimen_model.params_path))
    # list_model.load_state_dict(torch.load(regimen_model.params_path + '.model'))

    # Make dataloaders and predict
    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
    dev_predictions = regimen_model.predict(probe_model, list_model, dev_dataloader)
    train_predictions = regimen_model.predict(probe_model, list_model, train_dataloader)
    if do_test:
        test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
        test_predictions = regimen_model.predict(
            probe_model, list_model, test_dataloader
        )

    # Make dataloaders and report
    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
    reporter_model(train_predictions, train_dataloader, TRAIN_STR)
    reporter_model(dev_predictions, dev_dataloader, DEV_STR)
    if do_test:
        test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
        reporter_model(test_predictions, test_dataloader, TEST_STR)


if __name__ == "__main__":
    run_yaml_experiment()
