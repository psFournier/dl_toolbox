#!/bin/bash
#PBS -N tensorboard
#PBS -l select=1:ncpus=1
#PBS -l walltime=10:00:00
 
module load python/3.7.2 
TENSO=/home/eh/fournip/expes/venv/bin/tensorboard
"${TENSO}" --logdir /work/OT/ai4usr/fournip/outputs/christchurch --port 8122 --host $HOSTNAME &
echo 'ssh -L localhost:8122:'$HOSTNAME':8122 fournip@hal.sis.cnes.fr' > /work/OT/ai4usr/fournip/outputs/tensorboard
sleep 10h
