# heatSolverFenicsX

A Python-based finite element solver for the heat equation problems using the FEniCSx ecosystem, including `dolfinx`, `ufl`, and PETSc-backed linear solvers. This solver is coupled with Openfoam as a part of the ParaSiF partitioned solver for Thermal-Fluid-Structure Interaction (TFSI) using the Multiscale Universal Interface (MUI).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Visualisation](#visualisation)

## Project Structure
```
heatSolverFenicsX
│
├──src
  ├── solver.py              # Main solver containing outer and inner loops
  ├── heatEquationFenics     # Definition of variational form of heat eqation, creation of function spaces and PETSc solvers
  ├── meshGeneration.py      # Creation of mesh topology
  ├── boundarys.py           # Various BC definitions, includuding CoupledBoundaries responsible for MUI coupling
  ├── input.py               # Responsible for parsing case files; case.json and solver.json.
  ├── output.py              # Responsible for exporting mesh and field data to xdmf format for ParaView
├── README.md
├── originalCode.py          # Orginal code written by Wendi Liu from which this work is derived.
└── spack.yaml               # Spack environment dependencies

```

## Requirements

The project depends on the **FEniCSx** finite element ecosystem and mui4py (MUI Python Wrappers). The code has been tested using Python 3.12.

Required packages:
- `scipy v1.16.3`
- `dolfinx v0.9.0`
- `ufl v2024.2.0`
- `basix v0.9.0`
- `mpi4py v4.1.1`
- `petsc4py v3.24.3`
- `numpy v2.3.5`
- `mui4py @ master`
  
---

# Installation

If you are using the ParaSiF framework follow the installation details there, this will install any required dependencies for heatSolverFEniCSx. Otherwise, the best way to get started is using [spack](https://spack.io/). Using the **spack.yaml** file run the following with an environment name of your choice.

```bash
spack env create <environment name> spack.yaml
spack env activate -p <environment name>
spack install   # this will take a while
```


---

## Usage

To use the solver, create a case directory called "solid", similar to Openfoam with the following structure:
```
├──solid
  ├── output/
  ├── input/
    ├── case.json          # Specifies case parameters (e.g mesh lengths, thermal conductivity....)
    └── solver.json        # Specifies solver parameters (e.g mesh resolution, timestep...)
```
The json files should have the following formats:

**case.json**
```json
{
    "lx": 0.2,        # mesh length in x-direction
    "ly": 1.0,        # mesh length in y-direction
    "lz": 0.01,       # mesh length in z-direction
    "kappa": 54,      # thermal conductivity
    "alpha": 0.003,   # thermal diffuisivity
    "initial_temp": 273.15,    # initial field temperature
    "left_bc_temp": 273.15,    # left boundary temp
    "right_bc_temp": 274.15   # right boundary temp
    # Note: top and bottom boundarys are currently assumed to be insualted walls (ZeroGradient)
}
```

**solver.json**
```json
{
    "end_time": 5,      # end time in seconds for simulation
    "deltaT": 0.05,     # timestep
    "poly_order": 2,    # polynomial order of FEM solution
    "nx": 20,           # mesh resolution in x-direction
    "ny": 20,           # mesh resolution in y-direction
    "nz": 1,            # mesh resolution in z-direction
    "coupled_boundary_type": "neumann",      # coupled boundary type, possible options are: neumann, dirichlet, linearInterpolation, none
    "inner_loop_iterations": 8,              # iteration of inner loop for strong coupling
    "write_interval": 1.0,              # how often (in seconds) to write field data to xdmf file.
    "spatial_sampler": "pseudo_n2_linear",   # optional, MUI sampler used to fetch values across the interface
    "spatial_sampler_params": {"h": 0.2}     # optional, arguments for that sampler
}
```

`spatial_sampler` defaults to `pseudo_n2_linear` (linear interpolation between the two nearest points), so it can be left out. The available samplers and their parameters are:

| Sampler | Parameters (default) |
|---|---|
| `pseudo_n2_linear` | `h` search radius (0.2) |
| `pseudo_nearest_neighbor` | `h` search radius (0.2) |
| `exact` | `tol` (MUI default) |
| `gauss` | `r`, `h` (required) |
| `shepard_quintic` | `r` (0.2) |
| `sph_quintic` | `r` (0.2) |
| `sum_quintic` | `r` (0.2) |
| `rbf` | `r` (0.3), `basis` (0, Gaussian), `pou_size` (50) |

Note: `rbf` returns NaN when the incoming points all lie on one line, which is the case when the fluid interface is a single row of faces (as in the one-cell-thick example cases). Use `pseudo_n2_linear` on this side for those cases. `nearest_neighbor` and `moving_average` are not offered because their mui4py wrappers are broken.

No sampler extrapolates, so values at the ends of the interface are the least accurate: the solid end points lie beyond the last fluid face centre. For a standalone example of `rbf`, see [07-pseudo-diffusion](https://github.com/MxUI/MUI-demo/tree/master/07-pseudo-diffusion) in MUI-demo.

Then run the heat solver from the directory above "solid" as below. Note that if coupled_boundary_type is not set to 'none' then a corresponding openfoam script is required.

```bash
export PYTHONPATH=<path_to_heatSolverFenicsX>/heatSolverFenicsX/src:$PYTHONPATH
mpirun -np y python -m solver
# OR
mpirun -np y <openfoam script > : -np x python -m solver
```

---

## Visualisation

Solutions can be visualised using Paraview by opening "output/fenicsx_solid_data.xdmf".
