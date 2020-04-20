# Application: Vlasov-Poisson solver

The `vlasov_poisson` application solves the Vlasov-Poisson problem with a low-storage Runge–Kutta scheme. In each Runge-Kutta stage, following 5 steps are performed:

* **step 1 + 2**: Compute the degrees of freedom of the charge density <img src="https://render.githubusercontent.com/render/math?math=\rho"> via integration over the velocity space and compute compute the right-hand side for the Poisson equation:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\rho(t,\vec{x})=1-\int f(t,\vec{x},\vec{v})\,\mathrm{d}v">
</p>

* **step 3**: Solve the Poisson equation for the potential <img src="https://render.githubusercontent.com/render/math?math=\phi">:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=-\nabla_{\vec{x}}^2\phi(t,x)=\rho(t,\vec{x})">
</p>


* **step 4**: Compute the electric field <img src="https://render.githubusercontent.com/render/math?math=\vec{E}"> from <img src="https://render.githubusercontent.com/render/math?math=\phi">:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\vec{E}(t,\vec{x})=-\nabla_{\vec{x}}\phi(t,\vec{x})">
</p>

* **step 5**: Apply the advection operator:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial\,f}{\partial\, t}  %2B \left(\begin{array}{c}\vec{v} \\-\vec{E}(t,\vec{x})\end{array}\right)\cdot\nabla f = 0">
</p>

**NOTE:** For more detail, we refer readers to Sections 1 and 6 in [Munch, Kormann, and Kronbichler 2020](../../../../wiki/Publications).

## Running the application and parameter input files

**NOTE:** The same instructions as in the `advection` application are valid here.

The application can be run as follows:
```bash
./vlasov_poisson A.json
```
where `A.json` a parameter file is that configures the solver and the case.

To view and generate a valid dummy input parameter file, one can write:
```bash
./vlasov_poisson --help hyperrectangle
```
