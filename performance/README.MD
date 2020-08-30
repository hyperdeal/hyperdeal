
```bash
array=($(ls cases2/*.json)); mpirun -np 40 ./operators_advection_01 "${array[@]}"
```

```bash
array=($(ls cases2/*.json)); likwid-mpirun -np 40 -f -g CACHES -m -O ./operators_advection_01 "${array[@]}"
```
