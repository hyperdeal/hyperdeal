import json
import argparse, os
import math
import subprocess
import shutil

cmd = """#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o node-{1}.out
#SBATCH -e node-{1}.e
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=munch@lnm.mw.tum.de
# Wall clock limit:
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pr83te
#
## #SBATCH --switches=4@24:00:00
#SBATCH --partition={2}
#Number of nodes and MPI tasks per node:
#SBATCH --nodes={0}
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#module list

#source ~/.bashrc
# lscpu

module unload mkl mpi.intel intel
module load intel/19.0 mkl/2019
module load gcc/9
module unload mpi.intel
module load mpi.intel/2019_gcc
module load cmake
module load slurm_setup

pwd

array=($(ls node{1}/*.json))

mpirun -np {3} ./advection \"${{array[@]}}\"
"""

def run_instance(dim_x, dim_v, degree, n, c):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/tokamak_all.json", 'r') as f:
       datastore = json.load(f)

    N = {"1":[8,6],"2":[12,8],"4":[16,12],"8":[24,16],"16":[32,24],"32":[48,32],"64":[64,48],"128":[96,64],"256":[128,96],"512":[192,128], "1024":[256,192], "2048":[384, 256], "3072" : [384, 384]}

    NN = N[str(n)]

    # make modifications
    datastore["General"]["DimX"]       = dim_x
    datastore["General"]["DimV"]       = dim_v
    datastore["General"]["DegreeX"]    = degree
    datastore["General"]["DegreeV"]    = degree
    datastore["General"]["PartitionX"] = NN[0]
    datastore["General"]["PartitionV"] = NN[1]

    datastore["Case"]["NRefinementsX"]       = int(c / 2) + c %2
    datastore["Case"]["NRefinementsV"]       = int(c / 2)

    # write data to output file
    with open("node%s/inputs%s.json" % (str(n).zfill(4), str(c).zfill(2)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def compute_grid(dim, s):
    return [s / dim, [2 if i < s %dim else 1 for i in range(0, dim)] ]


def compute_grid_pair(dim_x, dim_v, s):
    return [compute_grid(dim_x, s / (dim_x + dim_v) * dim_x + min(s % (dim_x + dim_v), dim_x)), 
            compute_grid(dim_v, s / (dim_x + dim_v) * dim_v + max(dim_x, s % (dim_x + dim_v)) - dim_x)]

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('dim_x', action="store", type=int)
    parser.add_argument('dim_v', action="store", type=int)
    parser.add_argument('degree', action="store", type=int)

    args = parser.parse_args()

    # parameters
    dim_x  = args.dim_x;
    dim_v  = args.dim_v;
    degree = args.degree;

    folder_name = "torus-all-%s-%s-%s" %(str(dim_x), str(dim_v), str(degree) )

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)

    shutil.copy("../../advection", ".")

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072]:

        if not os.path.exists("node%s" % (str(n).zfill(4))):
            os.mkdir("node%s" % (str(n).zfill(4)))

        label = ""
        if n <= 16:
            label = "test"
        elif n <= 768:
            label = "general"
        elif n <= 3072:
            label = "large"

        with open("node%s.cmd" % (str(n).zfill(4)), 'w') as f:
            f.write(cmd.format(str(n), str(n).zfill(4), label, 48*n))
        

        print n
        for c in range(0,20):
            if 4 * n*48 <= (30 * 32) * 2**c:
                run_instance(dim_x, dim_v, degree, n, c)


if __name__== "__main__":
  main()

