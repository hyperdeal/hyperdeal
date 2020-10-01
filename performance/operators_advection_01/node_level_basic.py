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
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pr83te
#
## #SBATCH --switches=4@24:00:00
#SBATCH --partition=micro
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

for K in 2 3 4 5
do

    if [ -d \"k$K\" ]; then
        array=($(ls k$K/*.json))
        likwid-mpirun -np 48 -f -g CACHES   -m -O ./operators_advection_01 \"${{array[@]}}\" | tee caches$K.o | tee -a caches.out
        likwid-mpirun -np 48 -f -g FLOPS_DP -m -O ./operators_advection_01 \"${{array[@]}}\" | tee flops$K.o  | tee -a flops.out
    fi
    
done

for K in 1 2 4 8
do

    if [ -d \"v$K\" ]; then
        array=($(ls v$K/*.json))
        likwid-mpirun -np 48 -f -g CACHES   -m -O ./operators_advection_01 \"${{array[@]}}\" | tee caches$K.o | tee -a caches.out
        likwid-mpirun -np 48 -f -g FLOPS_DP -m -O ./operators_advection_01 \"${{array[@]}}\" | tee flops$K.o  | tee -a flops.out
    fi
    
done


"""

def run_instance(type, v, k, dim_x, dim_v, s, do_collocation, do_ecl, detail):
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
    if type == 0:
        with open("k" + str(k) + "/" + str(dim_x + dim_v) + ".json", 'w') as f:
            json.dump(datastore, f, indent=4, separators=(',', ': '))
    else:
        with open("v" + str(v) + "/" + str(dim_x + dim_v) + ".json", 'w') as f:
            json.dump(datastore, f, indent=4, separators=(',', ': '))


def main():

    # select configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', help='foo help', action="store_true")
    parser.add_argument('--cache', help='foo help', action="store_true")
    parser.add_argument('--co', help='foo help', action="store_true")
    parser.add_argument('--fcl', help='foo help', action="store_true")
    parser.add_argument('--simd3', help='foo help', action="store_true")
    parser.add_argument('--simd4', help='foo help', action="store_true")
    args = parser.parse_args()
    
    # make sure that only one configuaration has been selected
    if((int(args.all) + int(args.cache) + int(args.co) + int(args.fcl) + int(args.simd3) + int(args.simd4) ) != 1):
        sys.exit("No configuration has been selected!") 
        
    do_collocation = False
    do_ecl         = True
    degrees        = [3, 5]
    vlens          = [0]
    detail         = False
        
    # set up configuration
    if(args.all):
        folder_name    = "node_level_basic_all"
        degrees        = range(2, 6)
    elif(args.cache):
        folder_name    = "node_level_basic_cache"
        detail         = True
    elif(args.co):
        folder_name    = "node_level_basic_co"
        do_collocation = True
    elif(args.fcl):
        folder_name    = "node_level_basic_fcl"
        do_ecl         = False
    elif(args.simd3):
        folder_name    = "node_level_basic_simd_k3"
        degrees        = [3]
        vlens          = [1, 2, 4, 8]
    elif(args.simd4):
        folder_name    = "node_level_basic_simd_k4"
        degrees        = [4]
        vlens          = [1, 2, 4, 8]
    else:
        sys.exit("No configuration has been selected!") 
    
    # create new folder for tests and switch to it
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)
    
    # copy executabel to folder
    shutil.copy("../../operators_advection_01", ".")
    
    type = 0
    
    if(len(vlens) == 1):
        type = 0
    elif(len(degrees) == 1):
        type = 1
    
    with open("job.cmd", 'w') as f:
        f.write(cmd.format())

    # loop over all degrees
    for vlen in vlens:
        for degree in degrees:

            # create folder for each degree
            if type == 0:
                if not os.path.exists("k" + str(degree)):
                    os.mkdir("k" + str(degree))
            if type == 1:
                if not os.path.exists("v" + str(vlen)):
                    os.mkdir("v" + str(vlen))

            # loop over all dimensions
            for dim in range(2,7):

                # set size limit of the problem
                limit = 1.0e9
                
                if dim == 2 and degree == 2:
                    limit = 0.1e9;
                elif degree == 2:
                    limit = 0.1e9;

                # decompose dimension
                dim_x = int(dim / 2 + dim % 2)
                dim_v = int(dim / 2)

                # determine a suitable mesh size
                s = [ i for i in range(1,100) if limit <= (degree+1)**dim * 2**i][0]

                # create json files
                run_instance(type, vlen, degree, dim_x, dim_v, util.compute_grid_pair(dim_x, dim_v, s), do_collocation, do_ecl, detail)


if __name__== "__main__":
  main()

