# scaled objective and constraints
# all constraints are computed w.r.t. "<=0", that is, c(x) - U or -c(x) + L
function fc!(nlp, P::PROBLEM, W::WORK, x)
    @inbounds @views begin
    W.f, W.mwork = objcons!(nlp, x[1:P.n], W.mwork)
    # upper bounds on "original" constraints, with bound in only
    # one side (all of the form c(x) <= U)
    @. W.uns_c[1:P.m] = W.mwork - P.cu
    if P.mLU > 0
        # duplicated constraints c(x) >= L from ones with both bounds
        @. W.uns_c[(P.m + 1):(P.m + P.mLU)] = W.mwork[P.cineqLU] - P.cl[P.cineqLU]
    end
    end
    # scale
    W.f *= P.wf
    @. W.c = P.wc * W.uns_c
    # adjust the signs of W.uns_c
    @. W.uns_c *= sign(P.wc)
    # complementarity slacks
    if P.ncc > 0
        ii = (P.n + 1):(P.n + P.ncc)
        W.uns_c[P.cc] .-= x[ii] ./ P.wc[P.cc]
        W.c[P.cc] .-= x[ii]
    end
    return
end

# scaled augmented Lagrangian
function augL(P::PROBLEM, W::WORK)
    aL = W.f
    @inbounds @views begin
    # "original" constraints
    @. W.mwork = W.c[1:P.m] + W.lam[1:P.m]/W.rho
    @. W.mwork[P.cineq] = max(0.0, W.mwork[P.cineq])
    aL += (0.5 * W.rho) * dot(W.mwork, W.mwork)

    # duplicated constraints
    if P.mLU > 0
        ii = (P.m + 1):(P.m + P.mLU)
        @. W.mwork[1:P.mLU] = max(0.0, W.c[ii] + W.lam[ii]/W.rho)
        aL += (0.5 * W.rho) * dot(W.mwork[1:P.mLU], W.mwork[1:P.mLU])
    end
    end
    return aL
end

# gradient of the scaled augmented Lagrangian. W.c must be relative to the given x
function gL!(nlp, gL, P::PROBLEM, W::WORK, x)
    ii = (P.m + 1):(P.m + P.mLU)
    @inbounds @views begin
    # factor for "original" constraints
    @. W.mwork = W.lam[1:P.m] + W.rho * W.c[1:P.m]
    @. W.mwork[P.cineq] = max(0.0, W.mwork[P.cineq])
    @. W.mwork *= P.wc[1:P.m]
    # sum factors of duplicated constraints
    if P.mLU > 0
        @. W.mwork[P.cineqLU] += P.wc[ii] *
            max(0.0, W.lam[ii] + W.rho * W.c[ii])
    end
    # add J'*(lam + rho*c), already considering duplicated constraints
    W.nwork = jtprod!(nlp, x[1:P.n], W.mwork, W.nwork)
    gL[1:P.n] .= W.nwork
    # add gradient of the objective function
    W.nwork = grad!(nlp, x[1:P.n], W.nwork)
    @. gL[1:P.n] += P.wf * W.nwork
    if P.ncc > 0
        # complementarity slacks
        gL[(P.n + 1):(P.n + P.ncc)] .= -W.mwork[P.cc]
    end
    end
    return
end

# p = proj_[xl,xu] (W.x - l*W.gL) - W.x
function minus_proj_lgL!(p, P::PROBLEM, W::WORK, l)
    @inbounds for i in 1:P.n
        diff = W.x[i] - l * W.gL[i]
        p[i] = ifelse(P.xl[i] <= diff,
            ifelse(diff <= P.xu[i], -l * W.gL[i], P.xu[i] - W.x[i]),
            P.xl[i] - W.x[i]
        )
    end
    @inbounds for i in (P.n + 1):(P.n + P.ncc)
        p[i] = -min(l * W.gL[i], W.x[i])
    end
    return
end

# compute the scaled optimality measure
function compute_opt(P::PROBLEM, W::WORK, epsfeas, nor; delta = 1e-6)
    if P.ncc == 0
        return norm(W.pgL, nor)
    else
        W.Lwork .= W.pgL
        @inbounds for j in 1:P.ncc
            a = P.cvar[j]
            b = P.n + j     # x[a] and x[b] are complementary vars
            if (W.x[a] <= delta)
                if (W.x[b] <= delta)
                    # x[a] = x[b] = 0
                    W.Lwork[a] = W.Lwork[b] = min(
                        abs(W.Lwork[a]),
                        abs(W.Lwork[b]),
                        max(max(0.0, W.Lwork[a]), max(0.0, W.Lwork[b]))
                    )
                else
                    # x[a] = 0, x[b] > 0
                    W.Lwork[a] = 0.0
                end
            else
                # x[a] > 0, x[b] = 0
                W.Lwork[b] = 0.0
            end
        end
        return norm(W.Lwork, nor)
    end
end

function compute_infeas_compl(P::PROBLEM, W::WORK)
    @inbounds @views begin
    res = norm(W.c[P.ceq], Inf)
    res = max(res, norm(min.(-W.c[P.cineq], W.lam[P.cineq] ./ W.rho), Inf))
    res = max(res, norm(min.(-W.c[(P.m + 1):(P.m + P.mLU)],
        W.lam[(P.m + 1):(P.m + P.mLU)] ./ W.rho), Inf))
    end
    return res
end

function compute_uns_infeas(P::PROBLEM, W::WORK)
    @inbounds @views begin
    res = norm(W.uns_c[P.ceq], Inf)
    res = max(res, norm(max.(0.0, W.uns_c[P.ceq]), Inf))
    res = max(res, norm(max.(0.0, W.uns_c[(P.m + 1):(P.m + P.mLU)]), Inf))
    end
    return res
end

function compute_opt_infeas(nlp, P::PROBLEM, W::WORK)
    @inbounds @views begin
    W.mwork[P.ceq] .= W.c[P.ceq]
    @. W.mwork[P.cineq] = max(0.0, W.c[P.cineq])
    @. W.mwork[P.cineqLU] += max(0.0, W.c[(P.m + 1):(P.m + P.mLU)])
    W.nwork = jtprod!(nlp, W.x[1:P.n], W.mwork, W.nwork)
    end
    return 0.5 * norm(W.nwork, Inf)
end
