module SimpleAL

using LinearAlgebra
using NLPModels
using Printf
using SparseArrays
using AmplNLReader
using Distances

include("utils.jl")
include("functions.jl")
include("output.jl")
include("spg.jl")

export al, std_extra_par, EXTRA_PAR

"""
    par = std_extra_par()

Initialize a structure containing extra parameters that
can be passed to the `al` function. All parameters are
initialized at their default values, but they can be
modified by setting `par.PARAMETER` to the desired value.

## General and AL parameters
- `tau`: reduction factor for infeasibility and complementarity
- `theta`: penalty parameter increase factor
- `lammin`: minimum allowable multiplier
- `lammax`: maximum allowable multiplier
- `rhomax`: maximum allowable penalty parameter
- `rhoinimin`: minimum allowable initial penalty parameter
- `rhoinimax`: maximum allowable initial penalty parameter
- `fmin`: threshold to declare problem unlimited
- `scale`: scale problem data before solving it?
- `wmin`: minimum scale factor
- `delta`: tolerance for declaring a complementary constraint equal to zero

## SPG/NPG/Line search parameters
- `eta`: Armijo's constant
- `lsm`: length of the non-monotone history
- `spg_maxit`: maximum number of iterations
- `t_redfac`: backtracking reduction factor (SPG)
- `sigma1`: minimum steplength to employ quadratic interpolation (SPG)
- `sigma2`: maximum factor to employ quadratic interpolation (SPG)
- `r_incfac`: backtracking increase factor (NPG)
- `lmin`: minimum allowable spectral steplength
- `lmax`: maximum allowable spectral steplength
- `epsnoprogf`: tolerance to declare a reduction in the objective
- `maxnoprogf`: maximum number of consecutive iterations without improvement in the objective
"""
function std_extra_par()
    return EXTRA_PAR(
        # AL extra parameters:
        0.8,        # tau
        5.0,        # theta
        -1e+20,     # lammin
        1e+20,      # lammax
        1e+20,      # rhomax
        1e-8,       # rhoinimin
        1e+8,       # rhoinimax
        -1e+20,     # fmin
        1e-6,       # delta
        # SPG parameters:
        1e-4,       # eta
        10,         # lsm
        10000,      # spg_maxit
        0.5,        # t_redfac
        0.1,        # sigma1
        0.9,        # sigma2
        1e-20,      # lmin
        1e+20,      # lmax
        5.0,        # r_incfac
        1e-1,       # epsnoprogf
        3000,       # maxnoprogf
        1e-8,       # wmin
        true        # false
    )
end

function initialize(nlp, extra_par::EXTRA_PAR, x0)
    # number of variables and constraints in the nlp structure
    n = nlp.meta.nvar
    m = nlp.meta.ncon

    # indices of constraints that are involved in complementarities
    cc = Int64[]
    cvar = Int64[]
    ncc = 0
    if typeof(nlp) == AmplModel
        # complementarities in nlp are of the form
        #   0 <= c(x) ⟂ y >= 0,
        # where y is a variable of the model.
        #
        # nlp.meta.cvar contains the indices of the variable y
        # for each constraint. If nlp.meta.cvar[j] == 0 then
        # the j-th constraint does not have a complementary var.
        # nlp.meta.n_cc is the number of complementary pairs.

        ncc = nlp.meta.n_cc
        if ncc > 0
            @inbounds for j in 1:m
                if nlp.meta.cvar[j] > 0
                    push!(cc, j)
                    push!(cvar, nlp.meta.cvar[j])
                end
            end
        end
    end

    # indices of equality, inequality constraints
    # original inequality constraints involved in complementarity becomes
    # equality
    ceq = Int64[]
    cineq = Int64[]
    cineqLU = Int64[]
    @inbounds for j in 1:m
        if (nlp.meta.lcon[j] == nlp.meta.ucon[j]) || (j in cc)
            push!(ceq, j)
        else
            push!(cineq, j)
            if !isinf(nlp.meta.lcon[j]) && !isinf(nlp.meta.ucon[j])
                push!(cineqLU, j)
            end
        end
    end
    mLU = length(cineqLU)

    # allocate workspace
    nn = n + ncc
    mm = m + mLU
    W = WORK(
        zeros(mm),                      # projected multipliers
        -Inf,                           # penalty
        Vector{Float64}(undef, nn),     # primal iterate
        Inf,                            # objective
        Vector{Float64}(undef, mm),     # scaled constraints
        Vector{Float64}(undef, mm),     # unscaled constraints
        Vector{Float64}(undef, nn),     # gradient of the augmented Lagrangian
        Vector{Float64}(undef, nn),     # (projected) grad. aug. Lag.
        Vector{Float64}(undef, n),      # working vector of size n
        Vector{Float64}(undef, m),      # working vector of size m
        Vector{Float64}(undef, nn),     # working vector for compute_opt
        # for SPG:
        Vector{Float64}(undef, nn),     # xnew
        Vector{Float64}(undef, nn),     # gLnew
        Vector{Float64}(undef, nn),     # s
        Vector{Float64}(undef, nn),     # y
        Vector{Float64}(undef, nn),     # xbest
        fill(-Inf, extra_par.lsm),      # lastL
        0.0                             # spectral steplength
    )

    # starting point
    W.x .= 0.0
    @inbounds if isempty(x0)
        if !isempty(nlp.meta.x0)
            W.x[1:n] .= nlp.meta.x0
        end
    else
        W.x[1:n] .= Float64.(x0[1:n])
    end

    # project the starting point
    @inbounds @views @. W.x[1:n] = clamp(
        W.x[1:n], nlp.meta.lvar, nlp.meta.uvar
    )

    # Adjust the starting point to satisfy complementarities:
    # if y = 0 (the original var in compl), make the corresponding slack = c[j]
    @inbounds if ncc > 0
        k = 0
        @views W.mwork = cons!(nlp, W.x[1:n], W.mwork)
        for j in 1:ncc
            if W.x[cvar[j]] == 0.0
                k += 1
                W.x[n + k] = max(0.0, W.mwork[j])
            end
        end
    end

    # compute objective function scaling factor
    # Obs: NLPModels already evaluate f considering the "min" sense
    if extra_par.scale
        @inbounds @views W.nwork = grad!(nlp, W.x[1:n], W.nwork)
        wf = max(extra_par.wmin, min(1.0, 1.0/norm(W.nwork, Inf)))
    else
        wf = 1.0
    end

    # compute constraints scaling factors
    wc = ones(m + mLU)
    @inbounds @views if extra_par.scale
        J = jac(nlp, W.x[1:n])
        for col in 1:(length(J.colptr) - 1)
            idx = J.colptr[col]:(J.colptr[col+1]-1)
            @. wc[J.rowval[idx]] =
                max(wc[J.rowval[idx]], abs(J.nzval[idx]))
        end
        @. wc[1:m] = max(extra_par.wmin, 1.0/wc[1:m])
        # duplicated constraints (those with both bounds) have the same
        # scaling factors, multiplied by -1 as they are of the form c(x) >= L
        @. wc[(m + 1):(m + mLU)] = -wc[cineqLU]
    end

    # transform c(x) >= L to -c(x) <= -L by adjusting the sign of wc
    # the sign w.r.t. duplicated constraints have already been set
    cl = deepcopy(nlp.meta.lcon)
    cu = deepcopy(nlp.meta.ucon)
    @inbounds for j in 1:m
        if !isinf(cl[j]) && isinf(cu[j])
            if !(j in cc)
                wc[j] *= -1.0
            end
            cu[j] = cl[j]  # sign is adjusted by the scale factor
            cl[j] = -Inf
        end
    end

    # allocate problem structure
    P = PROBLEM(
        n,                          # n in nlp
        m,                          # m in nlp
        cl,                         # cl
        cu,                         # cu
        deepcopy(nlp.meta.lvar),    # xl (does not contain compl slacks)
        deepcopy(nlp.meta.uvar),    # xu (does not contain compl slacks)
        consec_range(ceq),          # indices eq cons in nlp
        consec_range(cineq),        # indices ineq cons in nlp
        consec_range(cineqLU),      # indices ineq cons in nlp with both bounds
        mLU,                        # number constraints with both bounds
        ncc,                        # number compl slacks
        consec_range(cc),           # indices cons in nlp involved in compl
        cvar,                       # indice compl var for each cons in cc
        wf,                         # obj scaling factor
        wc                          # constraints scaling factors
    )

    return P, W
end

"""
    info = al(nlp; [PARAMETERS])

A simple safeguarded augmented Lagrangian method for
solving MPCCs as described in

Andreani, Rosa, Secchin. On the Boundedness of Multipliers
in Augmented Lagrangian Methods for MPCCs (submitted). 2025

`nlp` can be an `NLPModel` or `AmplModel` structure. If
an `NLPModel` object is passed, the model will be treated
as a standard NLP. If an `AmplModel` is passed, the
Lagrangian method where complementarity constraints
is maintained in the subproblems are employed.

## Properties in `info`
- `status`
  - = 0: approximate KKT point found
  - = 1: stationary point of the infeasibility
  - = 2: penalty parameter too large
  - = 3: maximum number of iterations reached
  - = 4: problem probably unlimited
  - =-1: unknown error
- `iter`: number of outer iterations performed
- `sol`: final primal point
- `f`: objective at `sol`
- `infeas`: sup-norm of constraint violation at `sol`
- `gL_supnorm`: sup-norm of the gradient of the Lagrangian at `sol`
- `infeas_compl`: constraints and complementarity violation at `sol`
- `lam`: Lagrange multipliers
- `max_lam_supnorm`: maximum sup-norm of multipliers during all the minimization process
- `final_rho`: last penalty parameter
- `time`: time in seconds

## Optional parameters
- `epsopt`: tolerance for optimality and complementarity (default = 1e-5)
- `epsfeas`: tolerance for feasibility (default = 1e-5)
- `maxit`: maximum number of outer iterations (default = 100)
- `verbose`: level of output information (0 to 2) (default = 1)
- `extra_par`: additional parameters, see the `std_extra_par` help
"""
function al(
    nlp;
    # AL parameters
    epsopt=1e-5,
    epsfeas=1e-5,
    maxit=100,
    verbose=1,
    x0=Float64[],
    # Extra parameters
    extra_par::EXTRA_PAR=std_extra_par(),
)
    assert_params(extra_par, nlp, epsfeas, epsopt, maxit)

    time = @elapsed begin

    # allocate and initialize fundamental structures
    P, W = initialize(nlp, extra_par, x0)

    # evaluate scaled objective and constraints
    fc!(nlp, P, W, W.x)

    # scaled feasibility + complementarity
    previous_infeas_compl = infeas_compl = compute_infeas_compl(P, W)

    # unscaled infeasibility
    infeas = compute_uns_infeas(P, W)

    opt_infeas = compute_opt_infeas(nlp, P, W)

    # norm of the projected gradient of aug L
    W.rho = 1.0
    gL!(nlp, W.gL, P, W, W.x)
    minus_proj_lgL!(W.pgL, P, W, 1.0)
    opt = compute_opt(P, W, epsfeas, Inf)

    # tolerance for the subproblem
    epsoptk = epsopt

    iter = 0
    status = -1

    # maximum proj_lam sup-norm over all iterations
    uns_maxnorm_lam = 0.0

    if verbose > 0
        banner(P, extra_par, epsopt, epsfeas, maxit)
    end

    # =============================
    # MAIN LOOP
    # =============================
    while (true)

        if verbose > 0
            al_print_info(P, W, iter, infeas, infeas_compl, opt, opt_infeas)
        end

        # =============================
        # Stopping criteria
        # =============================
        # AKKT point
        if (max(infeas, infeas_compl) <= epsfeas) && (opt <= epsopt)
            if verbose > 0
                println("\n EXIT: An approximate KKT point was found")
            end
            status = 0
            break
        end
        # Stationary point of the infeasibility
        if (infeas > sqrt(epsfeas)) &&
            (opt_infeas <= max(1e-12, 1e-4 * epsopt))
            if verbose > 0
                println("\n EXIT: Stationary point of the infeasibility")
            end
            status = 1
            break
        end
        # Too large penalty
        if (W.rho > extra_par.rhomax)
            if verbose > 0
                println("\n EXIT: Penalty parameter too large")
            end
            status = 2
            break
        end
        # Max iterations
        if (iter >= maxit)
            if verbose > 0
                println("\n EXIT: Maximum number of iterations reached")
            end
            status = 3
            break
        end
        # Objective << -1
        if (W.f < extra_par.fmin) && (infeas <= epsfeas)
            if verbose > 0
                println("\n EXIT: Objective decreased a lot... Is the " *
                    "problem unlimited?")
            end
            status = 4
            break
        end

        # start new iteration
        iter += 1

        # =============================
        # SUBPROBLEM
        # =============================
        # set penalty parameter
        if (iter <= 2)
            # ||c(x)||_2^2
            @inbounds @views W.mwork .= W.c[1:P.m]
            @inbounds @views @. W.mwork[P.cineq] = max(0.0, W.mwork[P.cineq])
            sq_infeas = dot(W.mwork, W.mwork)
            if P.mLU > 0
                @views @. W.mwork[1:P.mLU] =
                    max(0.0, W.c[(P.m + 1):(P.m + P.mLU)])
                @views sq_infeas += dot(W.mwork[1:P.mLU], W.mwork[1:P.mLU])
            end

            W.rho = clamp(
                2.0 * max(1.0, abs(W.f)) / max(1.0, sq_infeas),
                extra_par.rhoinimin, extra_par.rhoinimax
            )
        elseif (max(infeas, infeas_compl) > epsfeas^1.5) &&
            (infeas_compl > extra_par.tau * previous_infeas_compl)
            W.rho *= extra_par.theta
        end

        # set tolerance for the subproblem
        if (iter > 1) &&
            (infeas_compl <= sqrt(epsfeas)) && (opt <= sqrt(epsopt))
            epsoptk = min(extra_par.tau * opt, 1e-1 * epsoptk)
            epsoptk = max(epsoptk, 1e-1 * epsopt)
        end

        spg_iter, spg_st = spg(nlp, P, W, extra_par, epsoptk, epsfeas, verbose)

        # =============================
        # Update for the next iteration
        # =============================
        # objective, constraints, gL and pgL at new x were computed by spg

        # scaled optimality
        opt = compute_opt(P, W, epsfeas, Inf, delta = extra_par.delta)

        # estimate multipliers
        @. W.lam += W.rho * W.c
        @inbounds @views @. W.lam[P.cineq] = max(0.0, W.lam[P.cineq])
        @inbounds @views @. W.lam[(P.m + 1):(P.m + P.mLU)] =
            max(0.0, W.lam[(P.m + 1):(P.m + P.mLU)])

        # maximum norm of unscaled lam
        uns_maxnorm_lam =
            max(uns_maxnorm_lam, norm((P.wc .* W.lam) ./ abs(P.wf), Inf))

        # new projected multipliers
        clamp!(W.lam, extra_par.lammin, extra_par.lammax)

        # compute scaled squared-infeasibility gradient norm
        opt_infeas = compute_opt_infeas(nlp, P, W)

        # scaled infeasibility + complementarity
        previous_infeas_compl = infeas_compl
        infeas_compl = compute_infeas_compl(P, W)

        # unscaled infeasibility
        infeas = compute_uns_infeas(P, W)
    end
    end

    # compute lambda w.r.t. the original problem
    @. W.lam = (P.wc * W.lam) / abs(P.wf)
    @inbounds @views W.lam[P.cineqLU] .+= W.lam[(P.m + 1):(P.m + P.mLU)]

    # return data
    # TODO: compute unscaled opt, opt_infeas and infeas_compl
    @inbounds info = INFO(
        status,
        iter,
        deepcopy(W.x[1:P.n]),
        nlp.meta.minimize ? W.f / P.wf : -W.f / P.wf,
        (status == 1) ? opt_infeas : infeas,
        opt,
        infeas_compl,
        deepcopy(W.lam[1:P.m]),
        uns_maxnorm_lam,
        W.rho,
        time
    )

    return info
end
end
