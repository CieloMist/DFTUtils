# ----------------------------------------------------------------------- #
# Imports
    # System
import os
import sys
import json
import subprocess

    # System +
import numpy as np

    # MACE
import warnings
import yaml
import logging

# ----------------------------------- #
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()

config_file_path = os.getcwd() + '/config.yaml'
train_mace(config_file_path)