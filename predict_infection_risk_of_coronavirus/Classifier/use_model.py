# amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
import pandas as pd
import numpy as np
import readFasta
import re
import sys
import joblib
import os
from collections import Counter

def GGAP(fastas, gap, ** kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#']
    patterns = []
    for aa1 in AA:
        for aa2 in AA:
            header.append(aa1 + aa2)
            patterns.append(aa1 + "[" + AA + "]" + "{" + str(gap) + "}" + aa2)
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        length = len(sequence)
        denominator = length - gap - 1
        for pattern in patterns:
            fre = len(re.findall(pattern, sequence)) / denominator
            code.append(fre)
        encodings.append(code)
    return encodings, header


kw = {'path': r"H_train.txt", 'order': 'ACDEFGHIKLMNPQRSTVWY'}

if __name__ == "__main__":
    fastas1 = readFasta.readFasta(r"test.fasta")
    result, header = GGAP(fastas1, 13, **kw)
    data1 = np.matrix(result[1:])[:, 1:]
    data2 = np.matrix(result[1:])

    # data2 = np.matrix(result)
    data_GGAP = pd.DataFrame(data=data1, columns=header[1:])
    data_GGAP_name = pd.DataFrame(data=data2, columns=header)
    data_GGAP.to_csv('GGAP_test_gap_13.csv')
    data_GGAP_name.to_csv('GGAP_test_name_gap_13.csv')
    X = pd.read_csv(
            "GGAP_test_gap_13.csv", index_col=0)

    loaded_rf = joblib.load("./random_forest_0.9873817034700315.joblib")
    print("Predict: " , loaded_rf.predict(X))
