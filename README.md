# SimpleAL

This is a simple Julia implementation of the safeguarded augmented Lagrangian method for solving nonlinear programming problems, including those with complementarity constraints, described in

[*Andreani, Rosa, Secchin. On the boundedness of multipliers in augmented Lagrangian methods for mathematical programs with complementarity constraints (submitted). 2025*](https://optimization-online.org/?p=31446)

## Installation

`]add https://github.com/leonardosecchin/SimpleAL.jl`

## Examples of use

### Solving a toy problem

Build the `NLPModels` model:

```
using JuMP, NLPModels, NLPModelsJuMP

P = Model()
@variable(P, x[1:2])
@objective(P, Min, (x[1] - 2)^2 + (x[2] - 1)^2)
@constraint(P, x[1] + x[2] <= 2)
@constraint(P, x[1]^2 - x[2] <= 0)

nlp = MathOptNLPModel(P)
```

Solves and stores the output information in `output`:

```
using SimpleAL

output = al(nlp)
```

`output` is an `INFO` structure that contains the output of the algorithm. For example `output.f` is the final objective value, `output.sol` is the final primal iterate, and so on. For a complete list of properties, run `?al`.


### Reading an AMPL `nl` file

The examples below solve the problem `scholtes5` from [MacMPEC](https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC). You must download the `scholtes.nl` file yourself.

Solves `scholtes5` by the specialized augmented Lagrangian (AL) method that maintains complementarities in the subproblem:

```
using SimpleAL
using AmplNLReader

mpcc = AmplModel("scholtes5.nl")

output = al(mpcc)
```

You can convert complementarities to regular constraints using the `AmplMPECModel` command. In this case, the standard AL method is applied, in which the complementarities are penalized:

```
mpcc = AmplModel("scholtes5.nl")
nlp = AmplMPECModel(mpcc)

output = al(nlp)
```

### Changing parameters

You can pass the main parameters of the method directly in the `al` function. For example,

```
output = al(mpcc, epsfeas=1e-4, epsopt=1e-4, maxit=50)
```

runs `al` modifying the tolerances for feasibility and optimality, and the maximum number of allowed iterations. Extra parameters can be passed via `extra_par` option:

```
# initializes the structure with extra default parameters
par = std_extra_par()

# changes the maximum number of SPG iterations
par.spg_maxit = 20000

# solves the problem
output = al(mpcc, extra_par=par)
```

For more details, run `?al` and `?std_extra_par`.


## Funding

This research was supported by the São Paulo Research Foundation (FAPESP) (grant 2024/12967-8) and the National Council for Scientific and Technological Development (CNPq) (grant 302520/2025-2), Brazil.


## How to cite

If you use this code in your publications, please cite us. For now, you can cite the preprint:

[*Andreani, Rosa, Secchin. On the boundedness of multipliers in augmented Lagrangian methods for mathematical programs with complementarity constraints (submitted). 2025*](https://optimization-online.org/?p=31446)
