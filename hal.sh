#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python

cd "${TMPDIR}"
mkdir DIGITANIE
#cities=( "Arcachon" "Biarritz" "Montpellier" "Toulouse" "Nantes" "Strasbourg" "Paris" )
cities=( "Toulouse" )
for city in "${cities[@]}" ; do
    #rsync -rv --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE/ DIGITANIE/
    rsync -rvL --include="${city}/" --include="COS9/" --include="*.tif" --exclude="*" /work/OT/ai4geo/DATA/DATASETS/DIGITANIE/ DIGITANIE/
done

/work/OT/ai4usr/fournip/venv/bin/python "${HOME}"/dl_toolbox/dl_toolbox/train/train_from_splitfile.py

module unload python
