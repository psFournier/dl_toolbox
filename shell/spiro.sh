#!/bin/bash
#SBATCH --job-name=test     # job's name
# --output = name of the output file  --error= name of error file (%j = jobID )
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=16                   # number of cores
#SBATCH --gres=gpu:1                # number of gpgpus
#SBATCH --time=00:00:02                # Wall Time
#SBATCH --mem-per-cpu=8G            # memory per core
#SBATCH --partition=gpu        # material partition 
#SBATCH --qos=co_long_gpu             # Need to specify QoS because it is not default QoS
##SBATCH --export=none              # Uncomment to start the job with a clean environnement and source of ~/.bashrc

HYDRA_FULL_ERROR=1 /stck/pfournie/venv/bin/python /scratchm/pfournie/dl_toolbox/dl_toolbox/train/train_hydra.py -m paths=spiro +experiment=run
