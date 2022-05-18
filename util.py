from doctest import IGNORE_EXCEPTION_DETAIL
from email.policy import default
from logging.handlers import RotatingFileHandler
from operator import index
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import random
import glob
import os, json
import re
import warnings
import statsmodels.api as sm
from statsmodels.formula.api import ols

def readNewData(f):
    """
    A utility function that creates a pandas dataframe given a filepath. This is the function
    used to read in the data from Flueckiger et al.'s dataset from the University of Basel.
    *parameters*
    f: a filepath
    """
    filename = f
    df = pd.read_csv(filename, engine='c')
    names = findIndicators("newDataSet")
    df = df[names]
    df = df.rename(columns={'ID': 'id'})
    df['id'] = df['id'] + 60
    df[df < 0] = np.nan
    df = df.groupby("id").mean().reset_index()
    df = df.set_index('id')
    return df

def createDataFrameCSV(f):
    """
    A utility function that creates a pandas dataframe given a filepath
    *parameters*
    f: a filepath to the file to be read in by pandas
    """
    filename = f
    names = findIndicators("education")
    df = pd.read_csv(filename, engine='c')
    df = df[names]
    df = df.set_index('id')
    return df

def createDataFrameJSON(f, names):
    """
    A utility function that creates a pandas dataframe from multiple JSON files in different directories.
    The directories and JSON files are obtained from Wang et al.'s dataset (Dartmouth University)
    *parameters*
    f: a filepath containing the relevant directories to be read into the dataframe
    names: a list of names containing the relevant indicators
    """
    dfs = []
    path = f

    file_list = glob.glob(os.path.join(path, '*json'))
    file_list.sort()

    for f in file_list:
        # extract id
        file = f.split("_")[-1]
        id = re.findall('[0-9]+', file)
        strings = [str(x) for x in id]
        a_string = "".join(strings)
        theID = int(a_string)
        
        data = pd.read_json(f)
        tmp = [theID]
        
        # check for bad frames
        flag = False
        for indicator in names:
            if indicator not in data:
                flag = True

        # df empty or does not contain indicator
        if data.empty or flag:
            for indicator in names:
                tmp.append(np.nan)
        

        # good to go: compute statistics for participant
        else:
            data = data[names]
            for indicator in names:
                tmp.append(data[indicator].mean(skipna=True))
        
        # updated df
        colNames = names.copy()
        colNames.insert(0, 'id')
        df = pd.DataFrame([tmp], columns = colNames)
        df = df.set_index('id')
        dfs.append(df)
        
    return pd.concat(dfs, axis=1)

def findIndicators(directory):
    """
    A utility function used to scrape Flueckiger et. al's (University of Basel) and Wang et al.'s 
    (Dartmouth University) datasets for their relevant indicators.
    *parameters*
    directory: a list of relevant indicators i.e variables of interest
    """
    if directory == "Class":
        return ["hours", "experience"] 
    elif directory == "Class 2":
        return ["grade"]
    elif directory == "Exercise":
        return ["exercise"]
    elif directory == "Sleep":
        return ["hour", "rate"]
    elif directory == "Stress":
        return ["level"]
    elif directory == "Social":
        return ["number"]
    elif directory == "education":
        return ["id", " gpa all"]
    else:
        return ["ID", "SQ", "PhysAct", "Exam", "HSG", "LGA"]
    return None

def mergeDataFrames(f):
    """
    The function used to produce a pandas dataframe from Wang et. al's data (Dartmouth University)
    """

    path = f
    directoriesJSON = [i for i in os.listdir(path) if "." not in i and i != "education"]
    warnings.filterwarnings('ignore')
    dfs = []
    for dir in directoriesJSON:
       names = findIndicators(dir)
       dfs.append(createDataFrameJSON(path + dir, names))

    # handle csv files
    dfs.append(createDataFrameCSV(path + "education/grades.csv"))

    theDF = pd.concat(dfs, axis=1)

    theDF = theDF.rename({' gpa all' : 'gpa', 
    'hour': 'sleep', 'rate': 'sleepRating', 'level': 'stress', 'hours': 'study', 
    'grade': 'expectedGrade', 'number': 'social'}, axis=1)


    # convert sleep to binary based on median avgSleep
    theDF = theDF.groupby(by=theDF.columns, axis=1).sum()

    # change interpretation of sleepRating from Student Life dataset
    theDF['sleepRating'].head(49).index = 4 - theDF['sleepRating'].head(49).index

    theDF['sleep'] = (theDF['sleep'] > theDF['sleep'].median())*1
    return theDF

def predictVariable(df, X, Z):
    """
    A function used to perform multiple imputation. This function was used in an attempt to fill in missing data
    for Wang et al.'s data (Dartmouth University).
    *parameters*
    df: a pandas dataframe
    X: the variable of interest where missing data will be predicted
    Z: a list of covariates that represent all potential confounders between
    the missing X and the observed X.
    """


    dfObserved = df[df[X] != 0]
    dfNotObserved = df[df[X] == 0]

    s = X + " ~  1  + " + " + ".join(Z)
    # generate model: assume outcome is continuous
    mod = sm.GLM.from_formula(
        formula=s, data=dfObserved, family=sm.families.Binomial()).fit()

    # move predictions column into original df
    predictions = mod.predict(dfNotObserved).to_frame()
    predictions.rename(columns={0: X}, inplace=True)
    theDF = pd.merge(df, predictions[X],
                     how='left', left_index=True, right_index=True)

    # reconfigure expectedGrade to include predictions for unobserved values
    theDF = theDF.groupby(by=theDF.columns, axis=1).sum()
    theDF[X] = theDF[X + '_x'] + theDF[X + '_y']
    theDF.drop([X + '_x', X + '_y'], axis=1, inplace=True)


    # resort columns lexicographically
    return theDF.reindex(sorted(theDF.columns), axis=1)


def main():
    """
    Implemented for testing purposes.
    """
    f = "/Users/jonathoncarl/s22/cs379/project/BaselData/academicSleep.csv"
    df = readNewData(f)
    # convert sleep quality from multinomial to binary
    df['SQ'] = (df['SQ'] > df['SQ'].median())*1

    # round LGA values up
    df['LGA'] = df['LGA'].apply(np.ceil)
    df['LGA'] = df['LGA'].apply(int)
    df = df.reset_index()
    del df['id']


    # process Dartmouth data and predict missing data via multiple imputation
    fDart = '/Users/jonathoncarl/s22/cs379/project/DartmouthData/'
    dfDartmouth = mergeDataFrames(fDart)
    dfDartmouth = dfDartmouth.reset_index()
    del dfDartmouth['id']
    dfDartmouth = predictVariable(dfDartmouth, "expectedGrade", ["study"])
    dfDartmouth = predictVariable(dfDartmouth, "gpa", ["study", "sleep", "stress"])
    df.to_csv('BaselDF.csv', index=False)
    dfDartmouth.to_csv('DartmouthDF.csv', index=False)

if __name__ == '__main__':
    main()
