import json
import argparse, os
import math
import subprocess
import shutil
import sys 

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

def run_instance(dim_x, dim_v, degree, n, c, s, mem, dist):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/weak.json", 'r') as f:
       datastore = json.load(f)

    N = {"1":[8,6],"2":[16,6],"4":[32,6],"8":[64,6],"16":[64,12],"32":[64,24],"64":[64,48],"128":[128,48],"256":[256,48],"512":[512,48], "1024":[512,96], "2048":[512, 192]}

    NN = N[str(n)]

    # make modifications
    datastore["General"]["DimX"]       = dim_x
    datastore["General"]["DimV"]       = dim_v
    datastore["General"]["DegreeX"]    = degree
    datastore["General"]["DegreeV"]    = degree
    datastore["General"]["PartitionX"] = NN[0]
    datastore["General"]["PartitionV"] = NN[1]

    datastore["Case"]["NRefinementsX"]       = s[0][0]
    datastore["Case"]["NSubdivisionsX"]["X"] = s[0][1][0]
    datastore["Case"]["NSubdivisionsX"]["Y"] = s[0][1][1]
    datastore["Case"]["NSubdivisionsX"]["Z"] = s[0][1][2]

    datastore["Case"]["NRefinementsV"]       = s[1][0]
    datastore["Case"]["NSubdivisionsV"]["X"] = s[1][1][0]
    datastore["Case"]["NSubdivisionsV"]["Y"] = s[1][1][1]
    datastore["Case"]["NSubdivisionsV"]["Z"] = s[1][1][2]
    
    if mem:
        datastore["Matrixfree"]["OverlappingLevel"] = 0
    else:
        datastore["Matrixfree"]["OverlappingLevel"] = 2
    
    if dist:
        datastore["TemporalDiscretization"]["PerformanceLogAllCalls"] = True
        datastore["TemporalDiscretization"]["PerformanceLogAllCallsPrefix"] = "node-%s" % str(n).zfill(4)
    else:
        datastore["TemporalDiscretization"]["PerformanceLogAllCalls"] = False

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
    parser.add_argument('--mem', help='foo help', action="store_true")
    parser.add_argument('--dist', help='foo help', action="store_true")
    args = parser.parse_args()

    if((int(args.mem) + int(args.dist)) > 1):
        sys.exit("Max one configuration can be selected!") 

    # parameters
    dim_x  = 3;
    dim_v  = 3;
    degree = 3;
    shift  = 18; # start with 8^6 cells, e.g., 32^6 dofs

    if args.mem:
        folder_name = "weak-%s-%s-%s-mem" %(str(dim_x), str(dim_v), str(degree) )
    elif args.dist:
        folder_name = "weak-%s-%s-%s-dist" %(str(dim_x), str(dim_v), str(degree) )
    else:
        folder_name = "weak-%s-%s-%s" %(str(dim_x), str(dim_v), str(degree) )

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)

    shutil.copy("../../advection", ".")

    for c, n in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]):

        if args.dist and n != 64:
            continue

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

        s = compute_grid_pair(dim_x, dim_v, c + shift)
        print s
        run_instance(dim_x, dim_v, degree, n, c + shift, s, args.mem, args.dist)

if __name__== "__main__":
  main()

