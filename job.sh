#!/bin/bash -x
#SBATCH -J Ibc-R
#SBATCH -o mesa.out11
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

module load mesasdk/230703
module load mesa/r24.03.1

cd ${SLURM_SUBMIT_DIR}

source $MESASDK_ROOT/bin/mesasdk_init.sh
export OMP_NUM_THREADS=8
###$(MESA_DIR)= '/cluster/software/mesa/r10398/mesasdk--190503'

./clean
./mk
./rn_all > run.txt
