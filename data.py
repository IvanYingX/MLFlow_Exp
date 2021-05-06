from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def get(args):
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=args.test_size)