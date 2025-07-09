# -*- coding: utf-8 -*-
"""

@author: sumaiya
"""


import math
import pandas as pd
from pprint import pprint
from collections import Counter


def entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])


def entropy_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instance = len(a_list) * 1.0
    probs = [x / num_instance for x in cnt.values()]
    return entropy(probs)


def info_gain(df, split, target, trace=0):
    df_split = df.groupby(split)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target: [entropy_list, lambda x: len(x) / nobs]})
    df_agg_ent.columns = ["entropy", "propObserved"]
    new_entropy = sum(df_agg_ent["entropy"] * df_agg_ent["propObserved"])
    old_entropy = entropy_list(df[target])
    return old_entropy - new_entropy


def id3(df, target, attribute_name, default_class=None):

    cnt = Counter(x for x in df[target])

    if len(cnt) == 1:
        return next(iter(cnt))

    elif df.empty or (not attribute_name):
        return default_class

    else:
        default_class = max(cnt.keys())
        gains = [info_gain(df, attr, target) for attr in attribute_name]
        index_max = gains.index(max(gains))
        best_attr = attribute_name[index_max]
        tree = {best_attr: {}}
        remaining_attr = [x for x in attribute_name if x != best_attr]

        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target, remaining_attr, default_class)
            tree[best_attr][attr_val] = subtree

        return tree


def classify(instance, tree, default=None):
    attribute = next(iter(tree))

    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classify(instance, result)
        else:
            return result
    else:
        return default


df_tennis = pd.read_csv('data2.csv')
print(df_tennis)
attribute_names = list(df_tennis.columns)
attribute_names.remove('PlayTennis')
tree = id3(df_tennis, 'PlayTennis', attribute_names)
print('\n\n The resultant decision tree is: \n\n')
pprint(tree)

