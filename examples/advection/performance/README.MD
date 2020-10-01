## Run all possible sizes

In folder:
```bash
>pwd
~/sw-hyperdeal-update/hyperdeal-build/examples/advection/performance
```

run:
```bash
# dim_x=3, dim_v=3, degree=3
python ~/sw-hyperdeal-update/hyperdeal/examples/advection/performance/all.py 3 3 3

# dim_x=2, dim_v=2, degree=5
python ~/sw-hyperdeal-update/hyperdeal/examples/advection/performance/all.py 2 2 5
```

**Note:** Don't forget to compile the executable with the right template arguments.


## Run weak scaling (with detail view and alternative partitioning)

In folder:
```bash
>pwd
~/sw-hyperdeal-update/hyperdeal-build/examples/advection/performance
```

run:
```bash
python ~/sw-hyperdeal-update/hyperdeal/examples/advection/performance/weak.py
```

run:
```bash
python ~/sw-hyperdeal-update/hyperdeal/examples/advection/performance/weak.py --mem
```

run:
```bash
python ~/sw-hyperdeal-update/hyperdeal/examples/advection/performance/weak.py --dist
```

