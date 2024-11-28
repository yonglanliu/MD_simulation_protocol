#!/bin/bash

#SBATCH --job-name=job_name
#SBATCH --partition=gpu
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:p100:3 
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --mem=10g


project="project_name"
system="system_name"
pdb="./ionized.pdb"
psf="./ionized.psf"
temperature=310
boxSize=15.1
boxShape="cubic"
n_gpu=3
simTime=3

module load openmm

if [ -f "./state/state_mini.xml" ]; then
	opt="md"
else
	opt="mini"
fi


if [ $opt == "mini" ]; then
	python utils.py --project $project --system $system
	python openmmSim.py --pdb $pdb --psf $psf --opt $opt --temperature $temperature --boxSize $boxSize --n_gpu $n_gpu --simTime $simTime >& mini.out
fi


if [ -f "./state/state_mini.xml" ]; then
	opt="md"
fi

python openmmSim.py --pdb $pdb --psf $psf --opt $opt --temperature $temperature --boxSize $boxSize --n_gpu $n_gpu --simTime $simTime >& md.out

