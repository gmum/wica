from itertools import permutations

import numpy as np
import scipy as sp
from pulp import *
from scipy.stats import spearmanr


def std(X):
    std_X = np.std(X, axis=0)
    std_X = np.where(std_X == 0., 1., std_X)
    return std_X


def normalize(X):
    mean_X = np.mean(X, axis=0)
    std_X = std(X)
    data = (X - mean_X) / std_X
    return data


def norm2(x):
    return np.sqrt(np.inner(x, x))


def abs_cosine(x1, x2):
    return np.abs(np.inner(x1, x2) / (norm2(x1) * norm2(x2)))


def max_corr(X, Y, maximum=True):
    """
    :param X: matrix data n x m
    :param Y: matrix data n x m
    :return: maximum mean correlation between X1 and X2
    """
    C = get_corr(X, Y)
    vals = []
    for perm in permutations(range(0, len(C))):
        value = 0
        for i in range(len(C)):
            value += np.abs(C[i][perm[i]])
        vals.append(value * 1.0 / len(C))
    if maximum:
        return np.max(vals)
    else:
        return np.min(vals)


def get_corr(X, Y):
    A = normalize(X)
    B = normalize(Y)
    C = np.hstack((A, B))
    n = X.shape[1]
    coeff = np.corrcoef(C.T)[n:, :n]
    C = coeff
    return C


def max_corr_aprox(X, Y):
    """
    This function is only for quick approximate evaluation purposes.
    """
    corr_xy = get_corr(X, Y)
    return np.mean(np.max(np.abs(corr_xy), 0))


def tucker_measure(X, Y):
    X = normalize(X)
    Y = normalize(Y)

    size = X.shape

    m11 = np.empty(size[1])
    m12 = np.empty(size[1])
    m1 = np.empty((2, size[1]))

    for i in range(size[1]):
        for j in range(size[1]):
            m11[j] = abs_cosine(X[:, i], Y[:, j])
            m12[j] = abs_cosine(Y[:, i], X[:, j])
        m1[0, i] = np.max(m11)
        m1[1, i] = np.max(m12)
    m1 = np.min(np.max(m1, axis=0))
    return m1


def spearman_matrix(X, Y=None, version="2"):
    if Y is None:
        rho, pval = spearmanr(X, b=Y)
        return 1 - np.abs(rho)
    else:
        if version == "1":
            res = np.reshape([spearmanr(X[:, i], Y[:, j])[0] for i in range(X.shape[1]) for j in range(Y.shape[1])],
                             (X.shape[1], Y.shape[1]))
            return 1 - np.abs(res)
        else:
            rho, pval = spearmanr(X, b=Y)
            return 1 - np.abs(rho[0:X.shape[1], X.shape[1]:])


def get_equality_constraints(X, Y, px, py):
    Ay = []
    for i in range(Y.shape[1]):
        A = np.zeros((X.shape[1], Y.shape[1]))
        A[:, i] = np.ones(X.shape[1])
        Ay.append(A.flatten())
    Ax = []
    for j in range(X.shape[1]):
        A = np.zeros((X.shape[1], Y.shape[1]))
        A[j, :] = np.ones(Y.shape[1])
        Ax.append(A.flatten())
    Aeq = np.concatenate((Ay, Ax), axis=0)
    beq = np.concatenate((py, px), axis=0)
    return Aeq, beq


def linear_prog(C_matrix, X, Y, px=None, py=None):
    px = px if px is not None else np.ones(X.shape[1], dtype=float) / X.shape[1]
    py = py if py is not None else np.ones(Y.shape[1], dtype=float) / Y.shape[1]
    Aeq, beq = get_equality_constraints(X, Y, px, py)
    C = C_matrix.flatten()
    res = sp.optimize.linprog(C, A_ub=None, b_ub=None, A_eq=Aeq, b_eq=beq)
    return res


def spearman_metric(X, Y):
    C_matrix = spearman_matrix(X, Y)
    res = linear_prog(C_matrix, X, Y)
    return res["fun"]


def il_prog(C_matrix, X, Y, px=None, py=None):
    px = px if px is not None else np.ones(X.shape[1], dtype=float)
    py = py if py is not None else np.ones(Y.shape[1], dtype=float)
    Aeq, beq = get_equality_constraints(X, Y, px, py)
    prob = LpProblem("IL_Spearman", LpMinimize)
    C = C_matrix.flatten()
    var = np.array([LpVariable("gamm_var_{}".format(i), 0, 1, LpInteger) for i in range(len(C))])
    prob += lpSum(C * var), "Total Cost"
    for i, (a, b) in enumerate(zip(Aeq, beq)):
        prob += LpConstraint(lpSum(a * var), sense=pulp.LpConstraintEQ, name='eq_con_{}'.format(i),
                             rhs=b)
    prob += LpConstraint(lpSum(var), sense=pulp.LpConstraintEQ, name='max_gamma_sum', rhs=X.shape[1])
    prob.writeLP("temp.lp")
    res = prob.solve()
    return res, prob


def spearman_metric_ilp(X, Y):
    C_matrix = spearman_matrix(X, Y)
    res, prob = il_prog(C_matrix, X, Y)
    return value(prob.objective) / X.shape[1]


def spearman_ilp(X, Y):
    C_matrix = spearman_matrix(X, Y)
    res, prob = il_prog(C_matrix, X, Y)
    vars_value = {v.name.split("_")[2]: v.varValue for v in prob.variables()}
    return value(prob.objective) / X.shape[1], vars_value
