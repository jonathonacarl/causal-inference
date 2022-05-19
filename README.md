# [The Effect of Sleep Quality on Academic Performance in Undergraduate Students](https://github.com/jonathonacarl/causal-inference/blob/main/paper/joncarl-paper.pdf)

## [A Causal Analysis of Sleep Quality on Academic Performance](https://github.com/jonathonacarl/causal-inference/blob/main/paper/joncarl-paper.pdf)

```diff
@@ This repository contains a few directories and files used to produce the results obtained in my paper @@
```


```diff
@@ See the following descriptions below for guidance in this repository. @@
```

### [paper](https://github.com/jonathonacarl/causal-inference/tree/main/paper)
```diff
@@ Contains a pdf of my final paper, along with a matplotlib graph summarizing my causal estimates and my posited ADMG for my sensitivity analysis. @@
```

### [util.py](https://github.com/jonathonacarl/causal-inference/blob/main/util.py)
```diff
@@ Includes utility functions that read in csv, json files into pandas dataframes. Also includes a function to perform multiple imputations for missing data. @@
```
### [estimators.py](https://github.com/jonathonacarl/causal-inference/blob/main/estimators.py)
```diff
@@ Includes a plethora of ACE estimators implemented by me! These estimators include Backdoor, IPW, AIPW, and Dual IPW. @@
```
### [inference.py](https://github.com/jonathonacarl/causal-inference/blob/main/inference.py)
```diff
@@ The file where causal inference is performed. I use Ananke estimators, as well as Dual IPW, to generate the estimates mentioned in my paper. @@
```
### [BaselData](https://github.com/jonathonacarl/causal-inference/tree/main/BaselData)
```diff
@@ Directory containing survey data for a cohort of 72 first year students at the University of Basel. This is the data I use for causal analysis in my paper. @@
```
### [DartmouthData](https://github.com/jonathonacarl/causal-inference/tree/main/DartmouthData)
```diff
@@ Directory containing survey data for a cohort of 48 students at Dartmouth University. This is the data I used for sensitivity analysis in my paper. @@
```
### [pdDFs](https://github.com/jonathonacarl/causal-inference/tree/main/pdDFs)
```diff
@@ Directory containing csv files for the dataframes I use for analysis in my paper. These dataframes are obtained from BaselData and DartmouthData, but have been processed for the purpose of analysis. See 3.1 Data Processing in my paper for more on my data transformation. The DFs are included for ease of reproducibility of causal estimates obtained in my paper. @@
```
### [tetrad](https://github.com/jonathonacarl/causal-inference/tree/main/tetrad)
```diff
@@ Directory containing the Tetrad application I used for graph elicitation and causal discovery in my paper. Also contains a pdf of the graph learned by Tetrad. Minor edge additions/deletions were made based on background knowledge to find the final ADMG used for causal estimation in my paper. See 4.1 Learned ADMG for more on these edge decisions. @@
```
