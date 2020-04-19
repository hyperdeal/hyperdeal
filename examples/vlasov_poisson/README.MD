# Application: Vlasov-Poisson solver

The `vlasov_poisson` application solves the Vlasov-Poisson problem in high dimensions.

**NOTE:** For more detail, we refer readers to Section 6 in Munch, Kormann, and Kronbichler 2020.

## Running the application and parameter input files

**NOTE:** The same instructions as in in the `advection` application is valid here.

Tha application can be run as follows:
```bash
./vlasov_poisson A.json
```
where `A.json` a parameter file is that configures the solver and the case.

To view and generate a valid dummy input parameter file, one can write:
```bash
./vlasov_poisson --help hyperrectangle
```
