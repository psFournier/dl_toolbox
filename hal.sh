#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2
cd "${TMPDIR}"

#cp -r "${USR_DIR}"/miniworld_tif/christchurch .
#cp -r "${USR_DIR}"/SemCity-Toulouse-bench .
#cp -r "${USRDIR}"/digitanie .
cp -r /work/OT/ai4geo/DATA/DATASETS/DIGITANIE .

"${HOME}"/ai4geo/venv/bin/python "${HOME}"/dl_toolbox/dl_toolbox/train/train.py

module unload python/3.7.2
