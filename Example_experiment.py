import argparse
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

parser = argparse.ArgumentParser(description='Predict breast cancer')
parser.add_argument('--model', type=str, help='The model to use', action='store')
parser.add_argument('--fold', type=int, help='Number of splits in the KFold', action='store')
parser.add_argument('--test_size', type=float, help='Size of the test samples (%)', action='store')
args = parser.parse_args()
print(args.model)

model_names = ['RandomForestClassifier', 'KFold', 'LogisticRegression', 'SGDClassifier', 
              'KNeighborsClassifier', 'AdaBoostClassifier']
model_list = [RandomForestClassifier(), KFold(), LogisticRegression(), SGDClassifier(), 
              KNeighborsClassifier(), AdaBoostClassifier()]
model_dict = {n:model for n, model in zip(model_names, model_list)}

clf = model_dict[args.model]

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

kf = KFold(n_splits=args.fold)

