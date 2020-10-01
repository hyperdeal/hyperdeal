import json
import argparse, os
import math
import subprocess
import shutil
import sys 

import util

cmd = """#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o job.out
#SBATCH -e job.e
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
#SBATCH --partition=test
#Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
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

module load likwid/4.3.3-perf

pwd

rm caches.out
rm flops.out

array=($(ls input/*.json))
mpirun -np 48 ./operators_advection_01 \"${{array[@]}}\"

#likwid-mpirun -np 48 -f -g CACHES   -m -O ./operators_advection_01 \"${{array[@]}}\" | tee caches.o | tee -a caches.out
#likwid-mpirun -np 48 -f -g FLOPS_DP -m -O ./operators_advection_01 \"${{array[@]}}\" | tee flops.o  | tee -a flops.out


"""

def run_instance(c, v, k, dim_x, dim_v, s, do_collocation, do_ecl, detail):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/node_level_basic.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["General"]["VLen"]       = v
    datastore["General"]["Dim"]        = dim_x + dim_v
    datastore["General"]["Degree"]     = k
    datastore["General"]["PartitionX"] = 8
    datastore["General"]["PartitionV"] = 6
    datastore["General"]["Details"]    = detail
    
    datastore["SpatialDiscretization"]["DoCollocation"] = do_collocation
    
    if(do_ecl == False) :
        datastore["MatrixFree"]["UseECL"]      = False
        datastore["MatrixFree"]["DoBuffering"] = True

    print s

    datastore["Case"]["NRefinementsX"]       = s[0][0]
    if dim_x>=1:
        datastore["Case"]["NSubdivisionsX"]["X"] = s[0][1][0]
    if dim_x>=2:
        datastore["Case"]["NSubdivisionsX"]["Y"] = s[0][1][1]
    if dim_x>=3:
        datastore["Case"]["NSubdivisionsX"]["Z"] = s[0][1][2]

    datastore["Case"]["NRefinementsV"]       = s[1][0]
    if dim_v>=1:
        datastore["Case"]["NSubdivisionsV"]["X"] = s[1][1][0]
    if dim_v>=2:
        datastore["Case"]["NSubdivisionsV"]["Y"] = s[1][1][1]
    if dim_v>=3:
        datastore["Case"]["NSubdivisionsV"]["Z"] = s[1][1][2]

    # write data to output file
    with open("input/" + str(str(c).zfill(2)) + ".json", 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))


def main():

    # select configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', help='foo help', action="store_true")
    args = parser.parse_args()
    
    # make sure that only one configuaration has been selected
    if((int(args.all)) != 1):
        sys.exit("No configuration has been selected!") 
        
    do_collocation = False
    do_ecl         = True
    degree         = 3
    vlen           = 0
    detail         = False
    dim            = 6
        
    # set up configuration
    if(args.all):
        folder_name    = "sizes"
    else:
        sys.exit("No configuration has been selected!") 
    
    # create new folder for tests and switch to it
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)
    
    # copy executabel to folder
    shutil.copy("../../operators_advection_01", ".")
    
    with open("job.cmd", 'w') as f:
        f.write(cmd.format())

    # loop over all degrees
    for c in range(8,40):
        if 4 * 48 <= 2**c:
            # create folder for each degree
            if not os.path.exists("input"):
                os.mkdir("input")

            # decompose dimension
            dim_x = int(dim / 2 + dim % 2)
            dim_v = int(dim / 2)

            # create json files
            run_instance(c, vlen, degree, dim_x, dim_v, util.compute_grid_pair(dim_x, dim_v, c), do_collocation, do_ecl, detail)


if __name__== "__main__":
  main()

