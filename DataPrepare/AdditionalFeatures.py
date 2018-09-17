import numpy as np
from pandas import pandas as pd
import os

#This function takes input files and inserts into them new columns of data.
#Newly inserted data is repeated in many rows based the on value of a column.
#Here the param 'colHeadToMatch' is the column to match in the input files.
#param 'matchingList' and 'dataForMatchingList' contain the list of values (to match)
#and the data corresponding to each of the value (that will inserted into the matching
#rows of the input files), respectively.
def insertRepeatedFeatures(inputdir, outputdir, colHeadToMatch, colHeadsToAdd, insertAtIndices,
                           matchingList, dataForMatchingList):
    n = len(colHeadsToAdd)
    dicts = [dict() for i in range(n)]
    colHeadSortedByInsertIndices = [x for _,x in sorted(zip(insertAtIndices, colHeadsToAdd))]
    
    i_rep = 0
    while i_rep < dataForMatchingList.shape[0]:
        curr = matchingList[i_rep]
        j = 0
        for d in dicts:
            d[curr] = int(dataForMatchingList[i_rep, j])
            j += 1
        i_rep = i_rep + 1
        
    files = [f for f in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir, f))]
    i = 0
    for f in files:
        df = pd.read_csv(os.path.join(inputdir, f), skiprows=0, index_col=0)
        print(f)
        
        j = 0
        for iad in sorted(insertAtIndices):
            df.insert(iad, colHeadSortedByInsertIndices[j], 1)
            j += 1
            
        j = 0
        for c in colHeadsToAdd:
            df.ix[:,[c]] = df.ix[:,[colHeadToMatch, c]].apply(lambda x: dicts[j][x[0].upper()], axis=1)
            j += 1
        df.to_csv(os.path.join(outputdir, f))  
