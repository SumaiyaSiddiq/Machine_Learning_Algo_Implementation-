# -*- coding: utf-8 -*-
"""


@author: sumaiya
"""

import numpy as np
import pandas as pd

data = pd.read_csv('data1.csv')
concepts = np.array(data.iloc[:, 0:-1])
print(concepts)
target = np.array(data.iloc[:, -1])
print(target)


def learn(concepts, target):

    specific_h = concepts[0].copy()
    print('initialization of specific_h and general_h')
    print(specific_h)

    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)

    for i, h in enumerate(concepts):
        if target[i] == 'yes':
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
                print(specific_h)
            print(specific_h)

        if target[i] == 'no':
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print('steps of candidate Elimation Algorithm ', i + 1)
        print(specific_h)
        print(general_h)

    indeces = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]

    for i in indeces:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print('----------------final answer----------------\n')
print('final specific_h: ', s_final, sep='\n')
print('final general_h: ', g_final, sep='\n')
