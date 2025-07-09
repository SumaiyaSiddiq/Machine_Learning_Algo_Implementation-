# -*- coding: utf-8 -*-
"""


@author: sumaiya
"""
'''
a = ["hi","hello"]
print(list(enumerate(a)))
'''

import numpy as np
import pandas as pd

data = pd.read_csv("enjoysport.csv")

concepts = np.array(data.iloc[:,0:-1])
print(concepts)

target = np.array(data.iloc[:,-1])
print(target)


















