#!/bin/bash
#SBATCH --job-name=test_gpu2022     # job's name
# --output = name of the output file  --error= name of error file (%j = jobID )
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=6                   # number of cores
#SBATCH --gres=gpu:1                # number of gpgpus
#SBATCH --time=01:00:00                # Wall Time
#SBATCH --mem-per-cpu=8G            # memory per core
#SBATCH --partition=gpu_a100        # material partition 
#SBATCH --account=ai4geo       # account (launch myaccounts to list your accounts) 
#SBATCH --qos=gpu_all               # Need to specify QoS because it is not default QoS
##SBATCH --export=none              # Uncomment to start the job with a clean environnement and source of ~/.bashrc

# to go to the submit directory 
cd ${SLURM_SUBMIT_DIR}
bash "${HOME}"/dl_toolbox/copy_resisc_to_node.sh
nvidia-smi >  output_$SLURM_JOBID.log
#HYDRA_FULL_ERROR=1 /work/AI4GEO/users/fournip/trex/bin/python "${HOME}"/dl_toolbox/dl_toolbox/experiments/train_hydra.py paths=trex paths.data_dir="${TMPDIR}" datamodule=resisc module=supervised module/network=efficientnet trainer=gpu datamodule.prop=80 module.network.weights=IMAGENET1K_V1 >> output_$SLURM_JOBID.log
HYDRA_FULL_ERROR=1 /work/AI4GEO/users/fournip/trex/bin/python "${HOME}"/dl_toolbox/dl_toolbox/experiments/train_hydra.py experiment=prop >> output_$SLURM_JOBID.log