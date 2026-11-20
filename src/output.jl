function al_print_info(P::PROBLEM, W::WORK, iter, infeas, snorm, opt, opt_infeas)
    if mod(iter,20) == 0
        println()
        println("      |         | unscaled |unscaled |  scaled | sc compl|  scaled |  scaled")
        println("   it |     rho |        f |  infeas |     opt | + infeas| ∇infeas | |lam|_∞")
        println(" ---------------------------------------------------------------------------")
    end
    if iter > 0
        @printf(" %4d | %7.1e | %8.1e | %7.1e | %7.1e | %7.1e | %7.1e | %7.1e\n", iter, W.rho, W.f / P.wf, infeas, opt, snorm, opt_infeas, norm(W.lam, Inf))
    else
        @printf(" %4d |         | %8.1e | %7.1e | %7.1e | %7.1e | %7.1e | %7.1e\n", iter, W.f / P.wf, infeas, opt, snorm, opt_infeas, norm(W.lam, Inf))
    end
end

@inline function spg_exit_st(st)
    if st == 0
        return "success!"
    elseif st == 3
        return "maximum number of iterations reached"
    elseif st == 4
        return "steplength too small... no progress can be expected"
    elseif st == 5
        return "lack of progress"
    else
        return "unknown error"
    end
end

function spg_print_info(W::WORK, iter, L, pgL_norm, final, st, verbose)
    if verbose > 2
        # print complete information
        if (mod(iter, 500) == 0) && !(final)
            println()
            println("  SPG it |          L |  |pj ∇L|_∞")
            println(" ---------------------------------")
        end
        if (mod(iter, 500) == 0) || (final)
            @printf(" %7d | %10.3e | %10.3e\n", iter, L, pgL_norm)
        end
        if final
            @printf("  SPG status: %s\n\n", spg_exit_st(st))
        end
    elseif final && (verbose == 2)
        # print summary
        @printf("\n       SPG status: %s\n       SPG summary: iter = %d, L = %10.3e, |proj ∇L|_∞ = %9.3e\n\n",
                spg_exit_st(st), iter, L, pgL_norm)
    end
end

function assert_params(par::EXTRA_PAR, nlp, epsfeas, epsopt, maxit)
    @assert nlp.meta.ncon > 0 "unconstrained problem"
    @assert epsfeas > 0 "epsfeas must be positive"
    @assert epsopt > 0 "epsopt must be positive"
    @assert maxit > 0 "maxit must be positive"
    @assert 0 <= par.tau < 1 "tau must be in [0,1)"
    @assert par.theta > 1 "theta must be > 1"
    @assert (par.lammax >= 0) && (par.lammin <= 0) "0 must be in [lammin,lammax]"
    @assert par.rhomax > 0 "rhomax must by positive"
    @assert 0 < par.rhoinimin <= par.rhoinimax "it must be 0 < rhoinimin <= rhoinimax"
    @assert par.delta > 0 "delta must be positive"

    @assert par.spg_maxit >= 1 "spg: maxit must be ≥ 1"
    @assert 0 < par.epsnoprogf <= 1 "spg: tol for declaring lack of progress must be in (0,1]"
    @assert par.maxnoprogf >= 1 "spg: maximum number of consecutive iter without progress must be ≥ 1"

    @assert 0 < par.eta < 1 "regular spg: Armijo parameter be positive in (0,1)"
    @assert par.lsm >= 1 "regular spg: lsm must be ≥ 1"
    @assert 0 < par.t_redfac < 1 "regular spg: reduction factor must be in (0,1)"
    @assert 0 < par.sigma1 < par.sigma2 < 1 "regular spg: it must be 0 < σ1 < σ2 < 1"
    @assert 0 < par.lmin <= par.lmax "regular spg: it must be 0 < lmin ≤ lmax"
    @assert par.r_incfac > 1 "compl spg: increase factor must be > 1"
    @assert 0 < par.wmin <= 1 "minimum scale factor must be in (0,1]"
end

function banner(P::PROBLEM, par::EXTRA_PAR, epsopt, epsfeas, maxit)
    println(" ",repeat('=',61))
    println(" This is a simple implementation of the safeguarded augmented")
    println(" Lagrangian method described in\n")
    println(" Andreani, Rosa, Secchin. On the Boundedness of Multipliers in")
    println(" Augmented Lagrangian Methods for MPCCs. SIOPT, 2027")
    println()
    println(" Visit https://github.com/leonardosecchin/almpcc for details")
    println(" ",repeat('=',61))
    if P.ncc > 0
        println(" The specialized AL method was chosen")
    else
        println(" The standard AL method was chosen")
    end
    println()
    println(" INPUT:")
    println(" ",repeat('-',48))
    @printf(" Total of variables                  = %9d\n", P.n)
    @printf(" Total of constraints                = %9d\n\n", P.m)
    @printf(" Variables with lower bound          = %9d\n", count(P.xl .> -Inf))
    @printf(" Variables with upper bound          = %9d\n", count(P.xu .< Inf))
    @printf(" Equality constraints                = %9d\n", length(P.ceq))
    @printf(" Inequality constraints              = %9d\n", length(P.cineq))
    @printf(" Ineq cons with bounds on both sides = %9d\n", length(P.cineqLU))
    if P.ncc > 0
        @printf(" Complementarity pairs               = %9d\n", P.ncc)
    end
    println()
    @printf(" Objective function scaling factor   = %9.2e\n", P.wf)
    @printf(" Smallest constraints scaling factor = %9.2e\n", minimum(abs.(P.wc)))
    println()
    println(" AL PARAMETERS:")
    println(" ",repeat('-',48))
    @printf(" Tolerance feasibility               = %9.2e\n", epsfeas)
    @printf(" Tolerance optimality                = %9.2e\n", epsopt)
    @printf(" Maximum number of iterations        = %9d\n", maxit)
    @printf(" Decrease feasibility factor         = %9.2e\n", par.tau)
    @printf(" Penalty increase factor             = %9.2e\n", par.theta)
    @printf(" Maximum penalty                     = %9.2e\n", par.rhomax)
    @printf(" lambda_min                          = %9.2e\n", par.lammin)
    @printf(" lambda_max                          = %9.2e\n", par.lammax)
    println()
    println(" INNER SOLVER PARAMETERS:")
    println(" ",repeat('-',48))
    @printf(" Maximum number of iterations        = %9d\n", par.spg_maxit)
    @printf(" Armijo's constant                   = %9.2e\n", par.eta)
    @printf(" Non-monotone history length         = %9d\n", par.lsm)
    @printf(" Steplength reduction factor         = %9.2e\n", par.t_redfac)
    @printf(" sigma1                              = %9.2e\n", par.sigma1)
    @printf(" sigma2                              = %9.2e\n", par.sigma2)
    @printf(" Maximum spectral steplength         = %9.2e\n", par.lmax)
    @printf(" Minimum spectral steplength         = %9.2e\n", par.lmin)
    println(" ",repeat('=',61))
end
