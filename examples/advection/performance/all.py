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

def run_instance(n, c, s):
    with open("all_ref.json", 'r') as f:
       datastore = json.load(f)

    N = {"1":[8,6],"2":[12,8],"4":[16,12],"8":[24,16],"16":[32,24],"32":[48,32],"64":[64,48],"128":[96,64],"256":[128,96],"512":[192,128], "1024":[256,192], "2048":[384, 256], "3072" : [384, 384]}

    NN = N[str(n)]

    # make modifications
    datastore["General"]["PartitionX"]  = NN[0]
    datastore["General"]["PartitionV"]  = NN[1]

    datastore["Case"]["NRefinementsX"]       = s[0][0]
    datastore["Case"]["NSubdivisionsX"]["X"] = s[0][1][0]
    datastore["Case"]["NSubdivisionsX"]["Y"] = s[0][1][1]
    datastore["Case"]["NSubdivisionsX"]["Z"] = s[0][1][2]

    datastore["Case"]["NRefinementsV"]       = s[1][0]
    datastore["Case"]["NSubdivisionsV"]["X"] = s[1][1][0]
    datastore["Case"]["NSubdivisionsV"]["Y"] = s[1][1][1]
    datastore["Case"]["NSubdivisionsV"]["Z"] = s[1][1][2]

    # write data to output file
    with open("all/node%s/inputs%s.json" % (str(n).zfill(4), str(c).zfill(2)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def compute_grid(dim, s):

    #return [s / dim, [[2,1][i < s %dim] for i in range(0, dim)] ]
    return [s / dim, [2 if i < s %dim else 1 for i in range(0, dim)] ]


def compute_grid_pair(dim_x, dim_v, s):
    return [compute_grid(dim_x, s / (dim_x + dim_v) * dim_x + min(s % (dim_x + dim_v), dim_x)), 
            compute_grid(dim_v, s / (dim_x + dim_v) * dim_v + max(dim_x, s % (dim_x + dim_v)) - dim_x)]

def main():

    # parameters
    dim_x = 3;
    dim_v = 3;

    if not os.path.exists("all"):
        os.mkdir("all")

    shutil.copy("../advection", "all")

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072]:

        if not os.path.exists("all/node%s" % (str(n).zfill(4))):
            os.mkdir("all/node%s" % (str(n).zfill(4)))

        label = ""
        if n <= 16:
            label = "test"
        elif n <= 768:
            label = "general"
        elif n <= 3072:
            label = "large"

        with open("all/node%s.cmd" % (str(n).zfill(4)), 'w') as f:
            f.write(cmd.format(str(n), str(n).zfill(4), label, 48*n))
        

        print n
        for c in range(8,40):
            if 4 * n*48 <= 2**c:
                s = compute_grid_pair(dim_x, dim_v, c)
                print s
                run_instance(n, c, s)


if __name__== "__main__":
  main()

