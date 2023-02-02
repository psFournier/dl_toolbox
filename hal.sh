#!/bin/bash
#PBS -q qgpgpudev
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=01:00:00

module load python
module load cuda/10.1
cd "${TMPDIR}"

#cp -r "${USR_DIR}"/miniworld_tif/christchurch .
#cp -r "${USR_DIR}"/SemCity-Toulouse-bench .
#cp -r "${USRDIR}"/digitanie .
#cp -r /work/OT/ai4geo/DATA/DATASETS/DIGITANIE .
#cp -r /work/OT/ai4usr/fournip/DIGITANIE .
#cp -r /work/OT/ai4usr/fournip/DIGITANIE/Arcachon .

"${HOME}"/dl_toolbox/venv/bin/python "${HOME}"/dl_toolbox/dl_toolbox/train/train_digitanie.py

module unload python
