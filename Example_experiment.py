import arguments
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

args = arguments.parse()

model_names = ['LogisticRegression',
               'SGDClassifier',
               'KNeighborsClassifier',
               'RandomForestClassifier',
               'AdaBoostClassifier']
assert args.model in model_names, f'Model not in the list \
    {" ".join(model_names)}'


X, y = load_breast_cancer(return_X_y=True)
X, X_test, y, y_test = train_test_split(X, y, test_size=args.test_size)


clf.fit(X, y)
print(pd.DataFrame(clf.cv_results_))

y_pred = clf.predict(X_test)
print(f'Accuracy on test set: {accuracy_score(y_test, y_pred)}')