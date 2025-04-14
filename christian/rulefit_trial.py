from imodels.rule_set.rule_fit import RuleFit
import pandas as pd
import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

pd.set_option("display.max_colwidth", 999)

data = preprocessing.clean_data("fraud.csv")

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RuleFit(n_estimators=50, tree_size=2, max_rules=10, random_state=0)

print("fitting the rulefit model")

rf.fit(X_train, y_train)

print("getting the rules")
rules = rf._get_rules()

print("printing the rules")
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)

rules.head()


""" features = X.columns
X = X.values

y_class = y.copy()
y_class[y_class < 21] = 0
y_class[y_class >= 21] = +1
N = X.shape[0]

#fit
rf = RuleFit(tree_size=4, sample_fract='default', max_rules=2000,
             memory_par=0.01,
             tree_generator=None,
             rfmode='classify', lin_trim_quantile=0.025,
             lin_standardise=True, exp_rand_tree_size=True, random_state=1,
             max_iter=100)    

rf.fit(X, y_class, feature_names=features)

#predict
y_pred = rf.predict(X)
y_proba = rf.predict_proba(X)

#basic checks for probabilities
assert np.min(y_proba) >= 0
assert np.max(y_proba) <= 1

#test that probabilities match actual predictions
np.testing.assert_array_equal(np.rint(np.array(y_proba[:,1])), y_pred)

cm = confusion_matrix(y, y_pred)
print(cm) """