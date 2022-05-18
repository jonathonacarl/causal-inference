## A Causal Analysis of Sleep Quality on Academic Performance

This repository contains a few directories and files used to produce the results obtained in my paper

### The Effect of Sleep Quality on Academic Performance in Undergraduate Students

See the following descriptions below for guidance in this repository.

# paper \\
Contains a pdf of my final paper, along with a matplotlib graph summarizing my causal estimates.

# util.py \\
Includes utility functions that read in csv, json files into pandas dataframes. Also includes a function to perform multiple imputations for missing data.

# estimators.py \\
Includes a plethora of ACE estimators implemented by me! These estimators include Backdoor, IPW, AIPW, and Dual IPW.

***inference.py***
The file where causal inference is performed. I use Ananke estimators, as well as Dual IPW, to generate the estimates mentioned in my paper.

***BaselData***
Directory containing survey data for a cohort of 72 first year students at the University of Basel. This is the data I use for causal analysis in my paper.

***DartmouthData***
Directory containing survey data for a cohort of 48 students at Dartmouth University. This is the data I used for sensitivity analysis in my paper.

***pdDFs***
Directory containing csv files for the dataframes I use for analysis in my paper. These dataframes are obtained from **BaselData** and **DartmouthData**, but have been processed for the purpose of analysis. See **3.1 Data Processing** in my paper for more on my data transformation. The DFs are included for ease of reproducibility of causal estimates obtained in my paper.
