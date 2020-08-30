## Comments on adding new tests

New tests can be added by providing following three files:
* `X.configuration`: containing relevant compile-time constants 
   (e.g. `DIM_X`, `DIM_V`, `DEGREE`, ...), which are used to build the 
   `advection` application
* `X.Y.json`: the input parameter file
* `X.Y.out`: the expected output

Example:
```bash
adv_1D_1D_k3.configuration
adv_1D_1D_k3.hyperrectangle_01.json
adv_1D_1D_k3.hyperrectangle_01.out
```

**NOTE:** Configuration files can be reused for multiple tests by using
the same prefix `X`.

## Description on existing tests

### adv_1D_1D_k3

Tensor product of two one dimension triangulations and `FE_DGQ<1+1>(3)`.

- hyperrectangle_01: test ECL + PBC
- hyperrectangle_03: test FCL + PBC

### adv_2D_2D_k3

Tensor product of two two dimension triangulations and `FE_DGQ<1+1>(3)`.

- hyperrectangle_01: test ECL + PBC
- hyperrectangle_02: test FCL + PBC
- hyperrectangle_03: test ECL + DBC
- hyperrectangle_04: test FCL + DBC
- hyperrectangle_05: test ECL + DBC + collocation 
- hyperrectangle_06: test FCL + DBC + collocation

### adv_2D_2D_k3_q5

Tensor product of two two dimension triangulations, `FE_DGQ<1+1>(3)`, and 
`QGauss<1>(5)` (over-integration):

- hyperrectangle_01: test ECL + PBC
- hyperrectangle_02: test FCL + PBC
- hyperrectangle_03: test ECL + DBC
- hyperrectangle_04: test FCL + DBC