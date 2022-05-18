# causal-inference
A Causal Analysis of Sleep Quality on Academic Performance

This repository contains a pdf (my paper) along with 3 python files used in generating the ACE estimates.

1. util.py: Includes utility functions that read in csv, json files into pandas dataframes. Also includes a function to perform multiple imputations for missing data.
2. estimators.py: Includes a plethora of ACE estimators implemented by me! These estimators include Backdoor, IPW, AIPW, and Dual IPW.
3. inference.py: The file where causal inference is performed. I use Ananke estimators, as well as Dual IPW, to generate the estimates mentioned in my paper.

The data used for analysis is also included for reproducibility purposes.
