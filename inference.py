from doctest import IGNORE_EXCEPTION_DETAIL
from email.policy import default
from hashlib import new
from logging.handlers import RotatingFileHandler
from operator import index
from time import sleep
from tkinter import font
from util import *
from estimators import *
from ananke.graphs import ADMG
from ananke.identification import OneLineID
from ananke.estimation import CausalEffect
from ananke.datasets import load_afixable_data
from ananke.estimation import AutomatedIF
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from fcit import fcit
import statistics
from math import sqrt

def main():
    """
    The layout of this function is as follows:
    1. Orient pd dataframe for analysis
    2. Causal Estimation via Primal IPW, Augmented Primal IPW, and Dual IPW
    3. Estimates visualized in matplotlib
    4. Sensitivity Analysis with Dartmouth Dataset
    """
    np.random.seed(0)
    
    ################################################
    # Orient pd dataframe for analysis
    ################################################

    f = "/Users/jonathoncarl/s22/cs379/project/BaselData/academicSleep.csv"
    dfbasel = readNewData(f)
    
    # convert sleep quality from multinomial to binary
    dfbasel['SQ'] = (dfbasel['SQ'] > dfbasel['SQ'].median())*1
    
    # round LGA values up
    dfbasel['LGA'] = dfbasel['LGA'].apply(np.ceil)
    dfbasel['LGA'] = dfbasel['LGA'].apply(int)
    dfbasel = dfbasel.reset_index()
    del dfbasel['id']

    ################################################
    # Causal Estimation: Primal IPW and Dual IPW
    ################################################
    
    # build ADMG learned in Figure 1
    di_edges = [('SQ', 'LGA'), ('PhysAct', 'LGA'), ('LGA', 'Exam'),
    ('HSG', 'LGA'), ('HSG', 'Exam')]

    bi_edges = [('SQ', 'PhysAct'), ('PhysAct', 'Exam'), ('SQ', 'Exam')]
    G = ADMG(dfbasel.columns, di_edges, bi_edges)

    ace_obj = CausalEffect(graph=G, treatment='SQ', outcome='Exam')
    
    # Primal IPW
    ace_pipw, Ql, Qu = ace_obj.compute_effect(
        dfbasel, "p-ipw", n_bootstraps=200, alpha=0.05)
    
    # Augmented Primal IPW
    ace_apipw, Ql2, Qu2 = ace_obj.compute_effect(
        dfbasel, "apipw", n_bootstraps=200, alpha=0.05)

    # Dual IPW (Ananke)
    ace_dipw, Ql3, Qu3 = ace_obj.compute_effect(
        dfbasel, "d-ipw", n_bootstraps=200, alpha=0.05)
    
    print("Primal IPW (Ananke) ACE: ", np.exp(ace_pipw),
          "(", np.exp(Ql), ", ", np.exp(Qu), ")")
    print("Augmented Primal IPW (Ananke) ACE: ", np.exp(ace_apipw),
          "(", np.exp(Ql2), ", ", np.exp(Qu2), ")")
    print("Dual IPW (Ananke) ACE: ", np.exp(ace_dipw),
          "(", np.exp(Ql3), ", ", np.exp(Qu3), ")")

    # Dual IPW (my own implementation)
    print("Dual IPW (own) ACE: ", np.exp(dual_ipw(data=dfbasel, Y="Exam", A="SQ",
        M="LGA", Z=["PhysAct", "HSG"])))
        # compute_confidence_intervals(Y="Exam", A="SQ", M="LGA", Z=["PhysAct", "HSG"],data=dfbasel,
        # method_name=[dual_ipw, False]) 
    
    ##################################
    ##      Generate Figure 2       ##
    ##################################
    plt.errorbar(["P-IPW"], [1.580], yerr=[[0.990], [2.711-1.580]], fmt='o', capsize=4)
    plt.errorbar(["APIPW"], [1.647], yerr=[[0.999], [3.823 - 1.647]], fmt='o', capsize=4)
    plt.errorbar(["D-IPW (Ananke)"], [1.647],
                 yerr=[[1.034], [4.090-1.647]], fmt='o', capsize=4)
    plt.errorbar(["D-IPW"], [1.103], yerr=[[0], [0]], fmt='o', capsize=4)
    plt.xlabel('Estimate Method', font="serif")
    plt.ylabel('ACE Estimate', font="serif")
    plt.text(0.05, 1.7, "1.580", font="serif")
    plt.text(1.1, 1.8, "1.647", font="serif")
    plt.text(2.1, 1.8, "1.647", font="serif")
    plt.text(2.8, 1.3, "1.103", font="serif")
    #plt.savefig("estimates.pdf")

    ################################################
    # Sensitivity Analysis via Dartmouth dataset
    ################################################

    # process Dartmouth data and predict missing data via multiple imputation
    fDart = '/Users/jonathoncarl/s22/cs379/project/DartmouthData/'
    dfdartmouth = mergeDataFrames(fDart)
    dfdartmouth = dfdartmouth.reset_index()
    del dfdartmouth['id']
    dfdartmouth = predictVariable(dfdartmouth, "expectedGrade", ["study"])
    dfdartmouth = predictVariable(dfdartmouth, "gpa", ["study", "sleep", "stress"])

    # Build posited ADMG discussed in Sensitivity Analysis
    di_edges2 = [('exercise', 'sleep'), ('exercise', 'stress'), ('social', 'stress'),
                 ('study', 'expectedGrade'), ('expectedGrade', 'gpa'), ('stress', 'gpa'), ('sleep', 'gpa')]

    bi_edges2 = [('study', 'sleep'), ('sleep', 'stress')]

    G2 = ADMG(dfdartmouth.columns, di_edges2, bi_edges2)

    # Augmented IPW estimate
    ace_obj2 = CausalEffect(graph=G2, treatment='sleep', outcome='gpa')
    ace_sens, Ql_sens, Qu_sens = ace_obj2.compute_effect(
        dfdartmouth, "aipw", n_bootstraps=200, alpha=0.05)
    print("AIPW ACE (Dartmouth): ", ace_sens, "(", Ql_sens, ",", Qu_sens, ")")


if __name__ == "__main__":
    main()
