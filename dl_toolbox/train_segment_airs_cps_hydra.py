import hydra
from omegaconf import DictConfig
import os
import sys
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig):
    working_dir = os.getcwd()
    

if __name__ == "__main__":
        
    run()
