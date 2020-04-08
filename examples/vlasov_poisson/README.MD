# Application: advection

The `advection` application performs a scalar transport in high dimensions.

The application can be run as follows:

```bash
mpirun -np X ./advection A.json B.json
```

A parameter file might look like this:

```json
{ 
    "General" : {
        "DimX"               : 2,
        "DimV"               : 2,
        "DegreeX"            : 3,
        "DegreeV"            : 3,
        "PartitionV"         : 1,
        "PartitionX"         : 1,
        "UseVirtualTopology" : true,
        "Verbose"            : true,
        "Case"               : "hyperrectangle"
    },
    "Case" : {
        "NRefinementsX"      : 2,
        "NRefinementsV"      : 2,
        "PeriodicX"          : true,
        "PeriodicV"          : true,
        "NSubdivisionsX"     : {"X" : 1, "Y" : 1, "Z" : 1},
        "NSubdivisionsV"     : {"X" : 1, "Y" : 1, "Z" : 1}
    },
    "Triangulation" : {
        "Type"               : "fullydistributed",
        "OutputGrid"         : false
    },
    "SpatialDiscretization" : {
        "DoCollocation"      : false,
        "MappingX"           : 1,
        "MappingV"           : 1
    },
    "TemporalDiscretization" : {
        "StartTime"          : 0.0,
        "FinalTime"          : 20.0,
        "TimeStep"           : 0.1,
        "MaxTimeStepNumber"  : 100000000,
        "CFLNumber"          : 0.3,
        "OutputTick"         : 0.2
    },
    "Matrixfree" : {
        "DoBuffering"        : false,
        "GhostFaces"         : true,
        "UseECL"             : true
    }
}
```