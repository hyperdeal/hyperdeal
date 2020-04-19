New tests can be added by providing following three files:
* `.configuration`: containing relevant compile-time constants 
   (e.g. `DIM_X`, `DIM_V`, `DEGREE`, ...), which are used to build the 
   `advection` application
* `.json`: the input parameter file
* `.out`: the expected output

Example:
```bash
adv_1D_1D_k3.configuration
adv_1D_1D_k3_hyperrectangle_01.json
adv_1D_1D_k3_hyperrectangle_01.out
```

**NOTE:** Configuration files can be reused for multiple tests by using
the same prefix.
