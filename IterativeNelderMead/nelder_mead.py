import numpy as np
from numba import njit

def get_default_options(options, p0, lower_bounds, upper_bounds, vary):
    nv = np.sum(vary)
    get_option(options, "max_fcalls", 1400 * nv)
    get_option(options, "no_improve_break", 3)
    get_option(options, "n_iterations", nv)
    get_option(options, "ftol_rel", 1E-6)
    get_option(options, "ftol_abs", 1E-10)
    get_option(options, "xtol_rel", 1E-12)
    get_option(options, "xtol_abs", 1E-16)
    get_option(options, "penalty", 10)

def get_option(options, key, default_value):
    if key not in options:
        options[key] = default_value

# Pn = (P - Pl) / Δ
# P = Pn * Δ + Pl
@njit
def normalize_parameters(values, lower_bounds, upper_bounds, vary):
    vn = np.zeros(len(values))
    lbn = np.zeros(len(values))
    ubn = np.zeros(len(values))
    for i in range(len(values)):
        vn[i], lbn[i], ubn[i] = normalize_parameter(values[i], lower_bounds[i], upper_bounds[i], vary[i])
    return vn, lbn, ubn

@njit
def normalize_parameter(value, lower_bound, upper_bound, vary):
    if np.isfinite(lower_bound) and np.isfinite(upper_bound) and lower_bound != upper_bound and vary:
        r = upper_bound - lower_bound
        vn = (value - lower_bound) / r
        return vn, 0.0, 1.0
    else:
        return value, lower_bound, upper_bound

@njit
def denormalize_parameters(valuesn, lower_bounds, upper_bounds, vary):
    v = np.zeros(len(valuesn))
    for i in range(len(valuesn)):
        v[i] = denormalize_parameter(valuesn[i], lower_bounds[i], upper_bounds[i], vary[i])
    return v

@njit
def denormalize_parameter(valuen, lower_bound, upper_bound, vary):
    if np.isfinite(lower_bound) and np.isfinite(upper_bound) and vary:
        r = upper_bound - lower_bound
        v = valuen * r + lower_bound
        return v
    else:
        return valuen

def get_subspaces(vary):
    subspaces = []
    vi = np.where(vary)[0]
    nv = len(vi)
    full_subspace = (None, vi, np.arange(nv).astype(int))
    if nv > 2:
        for i in range(nv-1):
            k1 = vi[i]
            k2 = vi[i+1]
            subspaces.append((i, [k1, k2], [i, i+1]))
        k1 = vi[1]
        k2 = vi[-1]
        subspaces.append((nv, [k1, k2], [1, nv]))
        if nv > 3:
            k1 = vi[2]
            k2 = vi[-2]
            subspaces.append((nv+1, [k1, k2], [2, nv-1]))
    return full_subspace, subspaces


def get_initial_simplex(p0n, lower_boundsn, upper_boundsn, vary):
    indsv = np.where(vary)[0]
    p0nv = p0n[indsv]
    nv = len(p0nv)
    simplex = np.tile(p0nv.reshape(nv, 1), (1, nv + 1))
    simplex[:, :-1] += np.diag(0.5 * p0nv)
    return simplex


def optimize(obj, p0, options=None, lower_bounds=None, upper_bounds=None, vary=None):

    if options is None:
        options = {}

    # Resolve Options
    n = len(p0)
    if vary is None:
        if lower_bounds is not None:
            vary = np.ones(n)
        else:
            vary = np.ones(n)
            for i in range(n):
                vary[i] = lower_bounds[i] != upper_bounds[i]
    if lower_bounds is None:
        lower_bounds = np.full(n, -np.inf)
    if upper_bounds is None:
        upper_bounds = np.full(n, -np.inf)
    get_default_options(options, p0, lower_bounds, upper_bounds, vary)

    # Normalize
    p0n, lower_boundsn, upper_boundsn = normalize_parameters(p0, lower_bounds, upper_bounds, vary)

    # Varied parameters
    vi = np.where(vary)[0]
    nv = len(vi)

    # If no parameters to optimize, return
    if nv == 0:
        fbest = obj(p0)
        return dict(pbest=p0, fbest=fbest, fcalls=0)

    # Number of iterations
    n_iterations = options["n_iterations"]

    # Subspaces
    full_subspace, subspaces = get_subspaces(vary)

    # Initial solution
    pbest = np.copy(p0)
    fbest = obj(p0)

    fbest_prev = fbest

    # Fcalls
    fcalls = 0

    # Full simplex
    full_simplex = get_initial_simplex(p0n, lower_boundsn, upper_boundsn, vary)

    current_iteration = 0

    state = {"fbest": fbest, "pbest": pbest, "fcalls": fcalls, "full_simplex": full_simplex}
    
    # Loop over iterations
    for iteration in range(n_iterations):

        current_iteration += 1

        # Perform Ameoba call for all parameters
        optimize_space(obj, state, full_subspace, p0, lower_bounds, upper_bounds, vary, np.copy(state["full_simplex"]), options)

        # Check x tolerance
        f_congerved = compute_df_rel(state["fbest"], fbest_prev) < options["ftol_rel"]
        if f_congerved:
            break

        fbest_prev = state["fbest"]
        
        # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
        if nv <= 2:
            break
        
        # Perform Ameoba call for subspaces
        for subspace in subspaces:
            pbestn, _, _ = normalize_parameters(state["pbest"], lower_bounds, upper_bounds, vary)
            initial_simplex = get_subspace_simplex(subspace, p0n, pbestn)
            optimize_space(obj, state, subspace, p0, lower_bounds, upper_bounds, vary, initial_simplex, options)
    
    # Output
    out = get_result(options, state, current_iteration, lower_bounds, upper_bounds, vary)

    # Return
    return out

@njit
def get_subspace_simplex(subspace, p0n, pbestn):
    n = len(subspace[1])
    simplex = np.zeros(n, n+1)
    simplex[:, 0] = p0n[subspace[1]]
    simplex[:, 1] = pbestn[subspace[1]]
    for i in range(1, n+2):
        simplex[:, i] = pbestn[subspace[1]]
        j = i - 2
        simplex[j, i] = p0n[j]
    return simplex

def optimize_space(obj, state, subspace, p0, lower_bounds, upper_bounds, vary, initial_simplex, options):
    
    # Simplex for this subspace
    simplex = np.copy(initial_simplex)
    nx, nxp1 = simplex.shape

    # Max f evals
    max_fcalls = options["max_fcalls"]
    ftol_rel = options["ftol_rel"]

    # Keeps track of the number of times the solver thinks it has converged in a row.
    no_improve_break = options["no_improve_break"]
    n_converged = 0

    # Penalty
    penalty = options["penalty"]

    # Initiate storage arrays
    fvals = np.zeros(nxp1)
    xr = np.zeros(nx)
    xbar = np.zeros(nx)
    xc = np.zeros(nx)
    xe = np.zeros(nx)
    xcc = np.zeros(nx)

    # Test parameters, normalized
    p0n, lower_boundsn, upper_boundsn = normalize_parameters(p0, lower_bounds, upper_bounds, vary)
    ptestn, _, _ = normalize_parameters(state["pbest"], lower_bounds, upper_bounds, vary)
    
    # Generate the fvals for the initial simplex
    for i in range(nxp1):
        fvals[i] = compute_obj(obj, simplex[:, i], state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty)

    # Sort the fvals and then simplex
    inds = np.argsort(fvals)
    simplex = simplex[:, inds]
    fvals = fvals[inds]
    x1 = simplex[:, 0]
    xn = simplex[:, -2]
    xnp1 = simplex[:, -1]
    f1 = fvals[0]
    fn = fvals[-2]
    fnp1 = fvals[-1]

    # Hyper parameters
    alpha = 1.0
    gamma = 2.0
    sigma = 0.5
    delta = 0.5
    
    # Loop
    while True:
            
        # Checks whether or not to shrink if all other checks "fail"
        shrink = False

        # break after max number function calls is reached.
        if state["fcalls"] >= max_fcalls:
            break
            
        # Break if f tolerance has been met no_improve_break times in a row
        if compute_df_rel(f1, fnp1) > ftol_rel:
            n_converged = 0
        else:
            n_converged += 1
        if n_converged >= no_improve_break:
            break

        # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
        # We want to iteratively replace the worst vector with a better vector.
        
        # The "average" vector, ignoring the worst point
        # We first anchor points off this average Vector
        xbar[:] = np.mean(simplex[:, 0:-1], axis=1)
        
        # The reflection point
        xr[:] = xbar + alpha * (xbar - xnp1)
        
        # Update the current testing parameter with xr
        fr = compute_obj(obj, xr, state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty)

        if fr < f1:
            xe[:] = xbar + gamma * (xbar - xnp1)
            fe = compute_obj(obj, xe, state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty)
            if fe < fr:
                simplex[:, -1] = xe
                fvals[-1] = fe
            else:
                simplex[:, -1] = xr
                fvals[-1] = fr
        elif fr < fn:
            simplex[:, -1] = xr
            fvals[-1] = fr
        else:
            if fr < fnp1:
                xc[:] = xbar + sigma * (xbar - xnp1)
                fc = compute_obj(obj, xc, state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty)
                if fc <= fr:
                    simplex[:, -1] = xc
                    fvals[-1] = fc
                else:
                    shrink = True
            else:
                xcc[:] = xbar + sigma * (xnp1 - xbar)
                fcc = compute_obj(obj, xcc, state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty)
                if fcc < fvals[-1]:
                    simplex[:, -1] = xcc
                    fvals[-1] = fcc
                else:
                    shrink = True
        if shrink:
            for j in range(1, nxp1):
                simplex[:, j] = x1 + delta * (simplex[:, j] - x1)
                fvals[j] = compute_obj(obj, simplex[:, j], state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty)

        # Sort
        inds = np.argsort(fvals)
        fvals = fvals[inds]
        simplex = simplex[:, inds]
        x1 = simplex[:, 0]
        xn = simplex[:, -2]
        xnp1 = simplex[:, -1]
        f1 = fvals[0]
        fn = fvals[-2]
        fnp1 = fvals[-1]


    # Sort
    inds = np.argsort(fvals)
    fvals = fvals[inds]
    simplex = simplex[:, inds]
    x1 = simplex[:, 0]
    xn = simplex[:, -2]
    xnp1 = simplex[:, -1]
    f1 = fvals[0]
    fn = fvals[-2]
    fnp1 = fvals[-1]
    
    # Update the full simplex and best fit parameters
    pbestn, _, _ = normalize_parameters(state["pbest"], lower_bounds, upper_bounds, vary)
    pbestn[subspace[1]] = x1
    vi = np.where(vary)[0]
    if subspace[0] is not None:
        state["full_simplex"][:, subspace[0]] = pbestn[vi]
    else:
        state["full_simplex"] = np.copy(simplex)

    # Denormalize and store
    pbest = denormalize_parameters(pbestn, lower_bounds, upper_bounds, vary)
    state["pbest"] = pbest

def get_result(options, state, current_iteration, lower_bounds, upper_bounds, vary):
    simplex_out = denormalize_simplex(state["full_simplex"], state["pbest"], lower_bounds, upper_bounds, vary)
    return dict(pbest=state["pbest"], fbest=state["fbest"], fcalls=state["fcalls"], simplex=simplex_out, iteration=current_iteration)

def denormalize_simplex(simplex, pars, lower_bounds, upper_bounds, vary):
    vi = np.where(vary)[0]
    ptempn = np.copy(pars)
    ptemp = np.copy(pars)
    simplex_out = np.zeros(simplex.shape)
    for i in range(len(vi)+1):
        ptempn[vi] = simplex[:, i]
        ptemp = denormalize_parameters(ptempn, lower_bounds, upper_bounds, vary)
        simplex_out[:, i] = ptemp[vi]
    return simplex_out


###################
#### TOLERANCE ####
###################

#@njit
def compute_df_rel(a, b):
    avg = (np.abs(a) + np.abs(b)) / 2
    return np.abs(a - b) / avg

#@njit
def compute_df_abs(a, b):
    return np.abs(a - b)



###########################################################################
###########################################################################
###########################################################################


def penalize(f, ptestn, indices, lower_boundsn, upper_boundsn, penalty):
    for i in range(len(indices)):
        j = indices[i]
        if ptestn[j] < lower_boundsn[j]:
            f += penalty * (lower_boundsn[j] - ptestn[j])
        if ptestn[j] > upper_boundsn[j]:
            f += penalty * (ptestn[j] - upper_boundsn[j])
    return f

def compute_obj(obj, x, state, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, penalty):
    state["fcalls"] += 1
    ptestn[subspace[1]] = x
    ptest = denormalize_parameters(ptestn, lower_bounds, upper_bounds, vary)
    f = obj(ptest)
    f = penalize(f, ptestn, subspace[1], lower_boundsn, upper_boundsn, penalty)
    if ~np.isfinite(f):
        f = 1E6
    return f