#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 0-01:00          # Maximum runtime in D-HH:MM
#SBATCH --mem=16G

#SBATCH -o log.out          # Write stdout to file named JOBIDNUM.out in current dir
#SBATCH -e log.err          # Write stderr to file named JOBIDNUM.err in current dir


module load class/default
module load cs137/2026spring
python -c "import torch; print(torch.__version__)"
