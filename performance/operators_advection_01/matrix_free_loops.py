import json
import argparse, os
import math
import subprocess
import shutil

from argparse import ArgumentParser

cmd = """#PBS -l nodes={0}:ppn=40
#PBS -l walltime=20:00:00
#PBS -N node{1}
#PBS -q gold
#PBS -j oe

echo -n "this script is running on: "
hostname -f
date

#export LD_LIBRARY_PATH="\${{LD_LIBRARY_PATH}}"

# since openmpi is compiled with PBS(Torque) support there is no need to
# specify the number of processes or a hostfile to mpirun.
env

cd {3}/all

pwd

#array=($(ls node{1}/*False_False_True_0.json));
#mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_0.tmp



#array=($(ls node{1}/*False_True_True_0.json));
#mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_1.tmp



#array=($(ls node{1}/*True_False_False_0.json));
#mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_2_0.tmp

#array=($(ls node{1}/*True_False_False_1.json));
#mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_2_1.tmp



array=($(ls node{1}/*True_True_True_0.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_3_0.tmp

array=($(ls node{1}/*True_True_True_1.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_3_1.tmp

array=($(ls node{1}/*True_True_True_2.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_3_2.tmp



array=($(ls node{1}/*True_True_False_0.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_4_0.tmp

array=($(ls node{1}/*True_True_False_1.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_4_1.tmp

array=($(ls node{1}/*True_True_False_2.json));
mpirun -np {2} ./operators_advection_01 \"${{array[@]}}\" | tee node{1}_4_2.tmp
"""

def run_instance(dim_x, dim_v, degree, n, c, s, sm, buffering, ecl, overlapping):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/matrix_free_loops.json", 'r') as f:
       datastore = json.load(f)

    N = {"1":[8,5],"2":[10,8],"4":[16,10],"8":[20,16]}

    NN = N[str(n)]

    # make modifications
    datastore["General"]["Dim"]     = dim_x + dim_v
    datastore["General"]["Degree"]     = degree
    datastore["General"]["PartitionX"] = NN[0]
    datastore["General"]["PartitionV"] = NN[1]

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
    
    datastore["General"]["UseSharedMemory"]     = sm
    datastore["MatrixFree"]["DoBuffering"]      = buffering
    datastore["MatrixFree"]["UseECL"]           = ecl
    datastore["MatrixFree"]["OverlappingLevel"] = overlapping

    # write data to output file
    with open("all/node%s/inputs%s_%s_%s_%s_%s.json" % (str(n).zfill(4), str(c).zfill(2), str(ecl), str(sm), str(buffering), str(overlapping)), 'w') as f:
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

    if not os.path.exists("all"):
        os.mkdir("all")

    shutil.copy("../operators_advection_01", "all")

    for n in [1, 2, 4, 8]:

        if not os.path.exists("all/node%s" % (str(n).zfill(4))):
            os.mkdir("all/node%s" % (str(n).zfill(4)))

        with open("all/node%s.cmd" % (str(n).zfill(4)), 'w') as f:
            f.write(cmd.format(str(n), str(n).zfill(4), 40*n, os.getcwd()))

        print n
        for c in range(8,40):
            if 8 * n*40 <= 2**c and (degree + 1)**(dim_x+dim_v) * 2**c <= 2e9 * n:
                s = compute_grid_pair(dim_x, dim_v, c)
                print s
                
                run_instance(dim_x, dim_v, degree, n, c, s, False, True,  False, 0)
                run_instance(dim_x, dim_v, degree, n, c, s, True,  True,  False, 0)
                
                for i in range(0,2):
                  run_instance(dim_x, dim_v, degree, n, c, s, False, False, True, i)
                  
                for i in range(0,3):
                  run_instance(dim_x, dim_v, degree, n, c, s, True,  True,  True, i)
                  
                for i in range(0,3):
                  run_instance(dim_x, dim_v, degree, n, c, s, True,  False, True, i)


if __name__== "__main__":
  main()
  