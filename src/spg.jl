function spg(nlp, P::PROBLEM, W::WORK, par::EXTRA_PAR, epsoptk, epsfeas, verbose)
    #W.x .+= 1e-6 * (1.0 .- 2.0*rand(length(W.x)))

    L = augL(P, W)

    fc!(nlp, P, W, W.x)
    gL!(nlp, W.gL, P, W, W.x)
    minus_proj_lgL!(W.pgL, P, W, 1.0)

    # reset history and save initial f
    W.lastL .= -Inf
    @inbounds W.lastL[1] = L

    opt = compute_opt(P, W, epsfeas, 2)

    # initial spectral steplength
    tsmall = max(1e-7 * norm(W.x, Inf), 1e-10)
    @. W.s = tsmall * W.pgL
    @. W.xnew = W.x + W.s
    gL!(nlp, W.gLnew, P, W, W.xnew)
    @. W.y = W.gLnew - W.gL
    sts = dot(W.s, W.s)
    sty = dot(W.s, W.y)
    W.spectral = (sty <= 0.0) ? par.lmax : clamp(sts/sty, par.lmin, par.lmax)

    # save initial solution as the best
    optbest = opt
    W.xbest .= W.x

    iter = 0
    status = 9
    Lprev = Inf
    bestredf = 0.0
    noprogf = 0

    # main loop
    while (true)
        spg_print_info(W, iter, L, opt, false, status, verbose)

        # test whether a solution was found
        if (max(opt, compute_opt(P, W, epsfeas, Inf)) <= epsoptk)
            status = 0
            spg_print_info(W, iter, L, opt, true, status, verbose)
            break
        end

        if (iter >= par.spg_maxit)
            W.x .= W.xbest
            fc!(nlp, P, W, W.x)
            gL!(nlp, W.gL, P, W, W.x)
            minus_proj_lgL!(W.pgL, P, W, 1.0)

            status = 3
            spg_print_info(W, iter, L, opt, true, status, verbose)
            break
        end

        # lack of progress
        currredf = Lprev - L
        bestredf = max(bestredf, currredf)
        if currredf <= par.epsnoprogf * bestredf
            noprogf += 1

            if noprogf >= par.maxnoprogf
                W.x .= W.xbest
                gL!(nlp, W.gL, P, W, W.x)
                minus_proj_lgL!(W.pgL, P, W, 1.0)
                fc!(nlp, P, W, W.x)

                status = 5
                spg_print_info(W, iter, L, opt, true, status, verbose)
                break
            end
        else
            noprogf = 0
        end

        # direction
        # W.pgL = proj_[xl,xu](x - lambda * gL) - x
        minus_proj_lgL!(W.pgL, P, W, W.spectral)

        Lprev = L

        if P.ncc == 0
            # usual SPG
            L, st = spg_ls(nlp, P, W, par, L, opt)
        else
            # NPG method of Guo and Deng
            L, st = spg_compl(nlp, P, W, par, L)
        end

        if isnan(L)
            W.x .= W.xbest
            fc!(nlp, P, W, W.x)
            gL!(nlp, W.gL, P, W, W.x)
            minus_proj_lgL!(W.pgL, P, W, 1.0)

            status = 4
            spg_print_info(W, iter, L, opt, true, status, verbose)
            break
        end

        # if no progress can be expected, returns the best x found so far
        if (st == 1)
            if iter > 0
                #W.x .= W.xbest
                W.xbest = W.x
                gL!(nlp, W.gL, P, W, W.x)
                minus_proj_lgL!(W.pgL, P, W, 1.0)
            end
            fc!(nlp, P, W, W.x)

            status = 4
            spg_print_info(W, iter, L, opt, true, status, verbose)
            break
        end

        # save the new L to the history
        @inbounds W.lastL[mod(iter + 1, par.lsm) + 1] = L

        # prepare for the next iteration
        # At this point, W.c and W.f are already relative to W.xnew
        gL!(nlp, W.gLnew, P, W, W.xnew)
        @. W.s = W.xnew - W.x
        @. W.y = W.gLnew - W.gL

        # update iterate
        W.x .= W.xnew
        W.gL .= W.gLnew

        minus_proj_lgL!(W.pgL, P, W, 1.0)
        opt = compute_opt(P, W, epsfeas, 2)

        # update spectral steplength
        sts = dot(W.s, W.s)
        sty = dot(W.s, W.y)

        W.spectral =
            (sty <= 0.0) ? par.lmax : clamp(sts/sty, par.lmin, par.lmax)

        # best iterate found so far
        if opt < optbest
            optbest = opt
            W.xbest .= W.x
        end

        iter += 1
    end

    return iter, status
end

# line search
function spg_ls(nlp, P::PROBLEM, W::WORK, par::EXTRA_PAR, L, pgL_norm)
    Lmax = maximum(W.lastL)

    # gL' * direction
    gLtd = dot(W.gL, W.pgL)

    # initial steplength
    t = 1.0

    @. W.xnew = W.x + W.pgL
    fc!(nlp, P, W, W.xnew)
    Lnew = augL(P, W)

    tmin = eps(norm(W.x, Inf))

    flag = 0
    while (Lnew > Lmax + t * par.eta * gLtd)
        if t <= tmin
            # t is too small, no progress can be expected
            @. W.xnew = W.x + tmin * W.pgL
            flag = 1
            break
        end

        if t <= par.sigma1
            t *= par.t_redfac
        else
            # quadratic interpolation
            tquad = -0.5 * ( gLtd * (t^2) / (Lnew - L - t * gLtd) )

            # backtracking
            t = ifelse((tquad < par.sigma1) || (tquad > par.sigma2 * t),
                t * par.t_redfac,
                tquad
            )
        end

        # new trial
        @. W.xnew = W.x + t * W.pgL

        fc!(nlp, P, W, W.xnew)
        Lnew = augL(P, W)
    end

    return Lnew, flag
end

# solve the subproblem with complementarities
function spg_compl(nlp, P::PROBLEM, W::WORK, par::EXTRA_PAR, L)
    Lmax = maximum(W.lastL)

    # initial penalty
    r = 1.0/W.spectral

    Lnew = L
    flag = 0
    xnorm2 = Inf
    while (Lnew > Lmax - 0.5 * r * par.eta * xnorm2)
        if r > 1.0/par.lmin
            # r is too large
            flag = 1
            break
        end

        @. W.xnew = W.x + (1.0/r) * W.pgL

        @inbounds for j in 1:P.ncc
            a = P.cvar[j]
            b = P.n + j     # x[a] and x[b] are complementary vars
            if W.xnew[a]^2 + min(0.0, W.xnew[b])^2 >=
                W.xnew[b]^2 + min(0.0, W.xnew[a])^2
                W.xnew[b] = 0.0
            else
                W.xnew[a] = 0.0
            end
        end

        fc!(nlp, P, W, W.xnew)
        Lnew = augL(P, W)
        @inbounds xnorm2 = sqeuclidean(W.xnew, W.x)
        r *= par.r_incfac
    end

    return Lnew, flag
end
