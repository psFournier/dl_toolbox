#!/bin/bash
#PBS -q qgpgpua30
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

bash "${HOME}"/dl_toolbox/copy_resisc_to_node.sh

module load python
/work/OT/ai4usr/fournip/latest/bin/python "${HOME}"/dl_toolbox/scripts/train_classif_resisc_cps.py
module unload python
