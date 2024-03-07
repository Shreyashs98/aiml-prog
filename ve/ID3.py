import pandas as pd
# id3
from math import log
from collections import Counter
def entropy(probs):
    return sum([-prob * log(prob, 2) for prob in probs])

def entropy_of_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)*1.0
    probs = [x/num_instances for x in cnt.values()]
    return entropy(probs)

def information_gain(df, split_attribute_name, target_attribute_name):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df) * 1.0
    df_agg_ent = df_split[target_attribute_name].agg([entropy_of_list, lambda x: len(x) / nobs])
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy

def id3(df, target_attribute_name, attribute_names):
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt)== 1:
        return next(iter(cnt))
    elif df.empty or(not attribute_names):
        return None
    else:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        best_attr = attribute_names[gainz.index(max(gainz))]
        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute_name, remaining_attribute_names)
            tree[best_attr][attr_val] = subtree
        return tree

df_tennis = pd.read_csv('playtennis.csv')
attribute_names = list(df_tennis.columns)
attribute_names.remove('PlayTennis')
tree = id3(df_tennis, 'PlayTennis', attribute_names)
print("\n\n The Resultant Decision Tree is:\n", tree)
