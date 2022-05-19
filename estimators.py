import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import mnlogit
import random
import math
from util import *

def odds_ratio(X, Y, Z, data):
    """
    Compute the odds ratio OR(X, Y | Z).
    *parameters*
    X, Y: names of variables in the data frame. 
    Z: list of covariates

    Return float OR for the odds ratio OR(X, Y | Z)
    """
    # for model
    s = str(X) + " ~ " + str(Y)

    # unpack covariates in Z
    for var in Z:
        s = s + " + " + str(var)

    # generate logistic regression model
    log_reg = sm.GLM.from_formula(
        formula=s, data=data, family=sm.families.Binomial()).fit()

    # OR = e^(beta_1) if X=Y=1, 1 otherwise
    return np.exp(log_reg.params[Y])


def backdoor_adjustment(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment
    *parameters*
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    """

    # build formula of form "Y | A, Z"
    s = str(Y) + " ~ " + str(A)

    # unpack covariates in Z
    for var in Z:
        s = s + " + " + str(var)

    # check for binary/continuous
    bin = data.columns[data.isin([0, 1]).all()]

    # arbitrarily assume binary
    mod_type = "binary"

    if Y not in bin:
        mod_type = "continuous"

    if mod_type == "binary":
        # logistic
        mod = sm.GLM.from_formula(formula=s, data=data,
                                  family=sm.families.Binomial()).fit()
    else:
        # continuous
        mod = sm.GLM.from_formula(formula=s, data=data,
                                  family=sm.families.Gaussian()).fit()

    # set A = 1
    data_a = data.copy()
    data_a[A] = 1

    # set A = 0
    data_a_prime = data.copy()
    data_a_prime[A] = 0

    # ACE
    return np.mean(mod.predict(data_a)) - np.mean(mod.predict(data_a_prime))


def ipw(Y, A, Z, data, trim=False):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via IPW
    *parameters*
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name of the treatment
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not
    """

    # build formula of form "A | Z" and find p(A=a | Z)

    if len(Z) != 0:
        ipwFormula = A + " ~ " + ' + '.join(Z)
        # binary treatment variable
        ipwModel = sm.GLM.from_formula(formula=ipwFormula, data=data,
                                       family=sm.families.Binomial()).fit()
    else:
        ipwModel = sm.GLM.from_formula(formula=A + " ~ 1", data=data,
                                       family=sm.families.Binomial()).fit()

    # add propensity scores
    propensity = ipwModel.predict(data)
    data = data.copy()
    data["propensity"] = propensity
    if trim:
        data = data[(data["propensity"] > 0.1) & (data["propensity"] < 0.9)]

    # indicator/propensity
    ipwAEqualsOne = (data[A])/propensity
    ipwAEqualsZero = (1-data[A])/(1-propensity)

    return np.mean((ipwAEqualsOne - ipwAEqualsZero)*data[Y])


def augmented_ipw(Y, A, Z, data, trim=False):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via AIPW
    *parameters*
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not
    """
    # build formula_ipw of form "A | Z" to find p(A | Z)
    if len(Z) != 0:
        ipwFormula = A + " ~ " + ' + '.join(Z)

        # binary treatment variable
        ipwModel = sm.GLM.from_formula(formula=ipwFormula, data=data,
                                       family=sm.families.Gaussian()).fit()
        backdoorFormula = Y + " ~ " + A + " + " + ' + '.join(Z)
    else:
        ipwModel = sm.GLM.from_formula(formula=A + " ~ 1", data=data,
                                       family=sm.families.Gaussian()).fit()
        backdoorFormula = Y + " ~ " + A

    # add propensity scores
    data = data.copy()
    propensity = ipwModel.predict(data)
    data["propensity"] = propensity

    if trim:
        data = data[(data["propensity"] > 0.1) & (data["propensity"] < 0.9)]

    #######
    # IPW #
    #######

    # binary case
    ipwAEqualsOne = data[A]/propensity
    ipwAEqualsZero = (1-data[A])/(1-propensity)

    #######################
    # Backdoor Adjustment #
    #######################

    backdoorModel = sm.GLM.from_formula(formula=backdoorFormula, data=data,
                                        family=sm.families.Gaussian()).fit()
    # set A = 1
    copyOne = data.copy()
    copyOne[A] = 1

    # set A = 0
    copyZero = data.copy()
    copyZero[A] = 0

    predictYOne = backdoorModel.predict(copyOne)
    predictYZero = backdoorModel.predict(copyZero)

    # E(Y | A=1 , Z) - # E(Y | A = 0 , Z)
    return np.mean(ipwAEqualsOne * (data[Y] - predictYOne) + predictYOne) - np.mean(ipwAEqualsZero * (data[Y] - predictYZero) + predictYZero)


def get_numpy_matrix(data, variables):
    """
    Returns a numpy matrix from a pandas dataframe and a list of variable names to
    be included in the matrix.
    *parameters*
    data: a pandas dataframe
    variable: a list of variable names to be included in the numpy matrix
    """

    matrix = data[variables].to_numpy()

    # if there's only one variable, ensure we return a matrix with one column
    # rather than just a column vector
    if len(variables) == 1:
        return matrix.reshape(len(data),)
    return matrix

def dual_ipw(data, Y, A, M, Z):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via Dual IPW
    *parameters*
    data: a pandas dataframe
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name of the treatment
    M: string corresponding variable name of the mediator
    Z: list of covariates used as the adjustment set
    """
    f = M + " ~ " + ' + '.join([A] + Z)
    # multinomial logistic treatment variable
    mod = mnlogit(formula=f, data=data).fit()

    prop = mod.predict(data)

    # create treatment datasets
    data_A0 = data.copy()
    data_A0[A] = 0
    prop_A0 = mod.predict(data_A0)

    data_A1 = data.copy()
    data_A1[A] = 1
    prop_A1 = mod.predict(data_A1)

    data["prop"] = np.ones(len(data))
    data_A0["propA0"] = np.ones(len(data))
    data_A1["propA1"] = np.ones(len(data))

    # fill in propensity scores for all 3 datasets
    for i in range(len(data)):

        mVal = data.iloc[i][M]
        data.at[i, 'prop'] = prop.at[i, mVal - 1]
        data_A0.at[i, 'propA0'] = prop_A0.at[i, mVal - 1]
        data_A1.at[i, 'propA1'] = prop_A1.at[i, mVal - 1]

    # ACE
    return np.mean((data_A1['propA1']/data['prop']) * data[Y]) - np.mean((data_A0['propA0']/data['prop']) * data[Y])


def compute_confidence_intervals(Y, A, M, Z, data, method_name, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for IPW or AIPW (potentially with trimming) via bootstrap.
    The input method_name can be used to decide how to compute the confidence intervals.
    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    *parameters*
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name of the treatment
    M: string corresponding variable name of the mediator
    Z: list of covariates used as the adjustment set
    method_name: estimation method used to obtain ACE
    num_bootstraps: number of iterations estimation method is performed
    alpha: significance level for confidence intervals
    """

    method = method_name[0]
    trimPropensity = method_name[1]

    estimates = []
    for i in range(num_bootstraps):
        # treat data as empirical distribution
        resample = data.sample(len(data), replace=True)
        resample = resample.reset_index()
        while (not resample[M].isin([1,2,3,4]).all()):
            resample = data.sample(len(data), replace=True)
            resample = resample.reset_index()
        # estimate and store resampled PEs
        if trimPropensity:
            resampled_pe = method(resample, Y, A, M, Z, trim=trimPropensity)
        else:
            resampled_pe = method(resample, Y, A, M, Z)
        estimates.append(resampled_pe)

    Ql = alpha/2
    Qu = 1 - alpha/2

    # pe will lie between q_low, q_up alpha % of time
    q_low, q_up = np.quantile(estimates, q=[Ql, Qu])

    return q_low, q_up

def main():
    """
    Implemented for testing purposes.
    """
    pass

if __name__ == '__main__':
        main()
