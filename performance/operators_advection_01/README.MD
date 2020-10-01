# Performence tests

## Test matrix_free_loops

This performance test the following matrix-free loop configurations:

| Configuration   | FCL | ECL | SM  | Buffering | Overlapping |
| --------------- |:---:|:---:|:---:|:---------:|:-----------:|
| Configuration 1 |  x  |     |     |    (x)    |     (0)     |
| Configuration 2 |  x  |     |  x  |     x     |     (0)     |
| Configuration 3 |     |  x  |     |    (x)    |     0/1     |
| Configuration 4 |     |  x  |  x  |     x     |    0/1/2    |
| Configuration 5 |     |  x  |  x  |           |    0/1/2    |

In total, 10 configurations are tested.


```bash
cd ~/sw-hyperdeal-update/hyperdeal-build-likwid/performance

mkdir operators_advection_01_results
cd operators_advection_01_results

# Cache analysis of advection operator (for 2<=k<=5, 2<=d<=6)
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_01/node_level_basic.py --all
cd node_level_basic_all/
sbatch job.cmd
cd ..

# Cache analysis of the cell-local part of the advection operator (for k=3/5, 2<=d<=6)
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_01/node_level_basic.py --cache
cd node_level_basic_cache/
sbatch job.cmd
cd ..

# Cache analysis of the cell-local part of the advection operator (for k=3/5, 2<=d<=6)
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_01/node_level_basic.py --fcl
cd node_level_basic_fcl/
sbatch job.cmd
cd ..

# Cache analysis of the cell-local part of the advection operator (for k=3/5, 2<=d<=6)
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_01/node_level_basic.py --co
cd node_level_basic_co/
sbatch job.cmd
cd ..

# Cache analysis of the cell-local part of the advection operator (for k=3/5, 2<=d<=6)
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_01/node_level_basic.py --simd3
cd node_level_basic_simd_k3/
sbatch job.cmd
cd ..

# Cache analysis of the cell-local part of the advection operator (for k=3/5, 2<=d<=6)
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_01/node_level_basic.py --simd4
cd node_level_basic_simd_k4/
sbatch job.cmd
cd ..



cd ..
mkdir operators_advection_02_results
cd operators_advection_02_results




# TODO
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_02/node_level_basic.py --cartesian
cd node_level_basic_simd_cartesian/
sbatch job.cmd
cd ..

# TODO
python ~/sw-hyperdeal-update/hyperdeal/performance/operators_advection_02/node_level_basic.py --full
cd node_level_basic_simd_full/
sbatch job.cmd
cd ..





```