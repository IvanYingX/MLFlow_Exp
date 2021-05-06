import arguments
import model
import data
import pandas as pd
from sklearn.metrics import accuracy_score

def run(clf, X, y):
    clf.fit(X, y)
    return clf

if __name__ == "main":
    args = arguments.parse()
    clf = model.choice(args)
    X, X_test, y, y_test = data.get(args)
    clf = run(clf, X, y)
    print(pd.DataFrame(clf.cv_results_))
    y_pred = clf.predict(X_test)
    print(f'Accuracy on test set: {accuracy_score(y_test, y_pred)}')
