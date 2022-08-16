#!/bin/bash

### Job name
#SBATCH --job-name=Bending_ohneBand_V9

### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=Bending_ohneBand_V9.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=48:00:00

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=2000

### Request number of tasks/MPI ranks
#SBATCH --ntasks=8

### Change to the work directory

### Execute the container
### myexecscript.sh contains all the commands that should be run inside the container
module load CONTAINERS
cd /home/nf838244/IEHK/DAMASK3/Biegesimulationen/64_ohneBand_V9
singularity exec /rwthfs/rz/SW/UTIL.common/singularity/damask-grid mpiexec -n 8 DAMASK_grid --load load.yaml --geom grid.vti

module load python/3.7.11

python3 damask3utilities.py

