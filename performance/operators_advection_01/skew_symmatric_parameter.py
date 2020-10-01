import json
import argparse, os
import math
import subprocess
import shutil

from argparse import ArgumentParser

cmd = """#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o node{1}.out
#SBATCH -e node{1}.e
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
#SBATCH --partition={4}
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

array=($(ls node{1}/*_0.0.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_0.0.out

array=($(ls node{1}/*_0.5.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_0.5.out

array=($(ls node{1}/*_1.0.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_1.0.out

"""

def run_instance(dim_x, dim_v, degree, n, c, s, beta):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/matrix_free_loops.json", 'r') as f:
       datastore = json.load(f)

    N = {"1":[8,6],"2":[12,8],"4":[16,12],"8":[24,16], "16" : [32, 24], "32" : [48, 32], "64" : [64, 48], "128" : [96, 64], "256" : [128, 96], "512" : [192, 128]}

    NN = N[str(n)]

    # make modifications
    datastore["General"]["Dim"]     = dim_x + dim_v
    datastore["General"]["Degree"]     = degree
    datastore["General"]["PartitionX"] = NN[0]
    datastore["General"]["PartitionV"] = NN[1]
    
    datastore["AdvectionOperation"]["SkewFactor"] = beta

    datastore["Case"]["NRefinementsX"]       = s[0][0]
    datastore["Case"]["NSubdivisionsX"]["X"] = s[0][1][0]
    datastore["Case"]["NSubdivisionsX"]["Y"] = s[0][1][1]
    if dim_x==3:
        datastore["Case"]["NSubdivisionsX"]["Z"] = s[0][1][2]

    datastore["Case"]["NRefinementsV"]       = s[1][0]
    datastore["Case"]["NSubdivisionsV"]["X"] = s[1][1][0]
    datastore["Case"]["NSubdivisionsV"]["Y"] = s[1][1][1]
    if dim_v==3:
        datastore["Case"]["NSubdivisionsV"]["Z"] = s[1][1][2]
    
    # write data to output file
    with open("./node%s/inputs%s_%s.json" % (str(n).zfill(4), str(c).zfill(2), beta), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def compute_grid(dim, s):
    return [s / dim, [2 if i < s %dim else 1 for i in range(0, dim)] ]


def compute_grid_pair(dim_x, dim_v, s):
    return [compute_grid(dim_x, s / (dim_x + dim_v) * dim_x + min(s % (dim_x + dim_v), dim_x)),
            compute_grid(dim_v, s / (dim_x + dim_v) * dim_v + max(dim_x, s % (dim_x + dim_v)) - dim_x)]
            

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")
    
    parser.add_argument('dim_x' , type = int, help="Dimension of x-space.")
    parser.add_argument('dim_v' , type = int, help="Dimension of v-space.")
    parser.add_argument('degree', type = int, help="Polynomial degree.")
    
    arguments = parser.parse_args()
    return arguments

def main():
    
    args = parseArguments()

    # parameters
    dim_x  = args.dim_x;
    dim_v  = args.dim_v;
    degree = args.degree;

    folder_name = "skew_symmatric_parameter"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name) 

    shutil.copy("../../operators_advection_01", ".")

    #nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    nodes = [1]

    for n in nodes:

        label = ""
        if n <= 16:
            label = "test"
        elif n <= 768:
            label = "general"
        elif n <= 3072:
            label = "large"

        if not os.path.exists("./node%s" % (str(n).zfill(4))):
            os.mkdir("./node%s" % (str(n).zfill(4)))

        with open("./node%s.cmd" % (str(n).zfill(4)), 'w') as f:
            f.write(cmd.format(str(n), str(n).zfill(4), 48*n, os.getcwd(), label))

        print n
        for c in range(8,40):

            if 4 * n*48 <= 2**c:
                s = compute_grid_pair(dim_x, dim_v, c)
                print s
                
                run_instance(dim_x, dim_v, degree, n, c, s, "0.0")
                run_instance(dim_x, dim_v, degree, n, c, s, "0.5")
                run_instance(dim_x, dim_v, degree, n, c, s, "1.0")


if __name__== "__main__":
  main()
  
