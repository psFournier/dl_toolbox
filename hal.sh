#!/bin/bash
#PBS -q qgpgpua30
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

#bash "${HOME}"/dl_toolbox/copy_semcity_to_node.sh

module load python
/work/OT/ai4usr/fournip/latest/bin/python "${HOME}"/dl_toolbox/dl_toolbox/train.py paths=hal trainer=gpu
module unload python
