#references: 
#discuss with b09902077
#for normalize: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
#for linear regression: https://medium.com/analytics-vidhya/implementing-gradient-descent-for-multi-linear-regression-from-scratch-3e31c114ae12
#for logisstic regression: https://medium.com/mlearning-ai/multiclass-logistic-regression-with-python-2ee861d5772a
#for decision tree classification: https://youtu.be/ZVR2Way4nwQ
#for random forest: https://youtu.be/v6VJ2RO66Ag

from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

labels_to_int = dict()
counter = -1


"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx] 
    test = df.iloc[split_idx:]

    # Split target and features
    #x->features, y->target
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    normalize_X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return normalize_X

#for writing report
def standardization (X: np.ndarray) -> np.ndarray:
    standardization_X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
    return standardization_X


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    global counter
    count_y = len(y)
    encode_labels_y = np.zeros(count_y)

    for i in range(count_y):
        if y[i] in labels_to_int:
            encode_labels_y[i] = labels_to_int[y[i]]
        else:
            counter+=1
            labels_to_int[y[i]]=counter
            encode_labels_y[i] = labels_to_int[y[i]]


    return encode_labels_y


def further_encode(unique_y, y: np.ndarray) -> np.ndarray:
    count_y = len(y)
    further_encode_y = np.zeros((count_y, unique_y))

    for i in range(count_y):
        further_encode_y[i][int(y[i])] = 1

    return further_encode_y
# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        # TODO: 2%
        if self.model_type == "logistic":
            self.w = np.ones((self.n_features, self.n_classes))
            new_y = further_encode(self.n_classes, y)
            self._compute_gradients(X,new_y)
        else:
            #linear
            self.w = np.zeros(self.n_features)
            self.b = 0
            self._compute_gradients(X,y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            predicted_value = np.dot(X, self.w) + self.b
            return predicted_value 

        elif self.model_type == "logistic":
            # TODO: 2%
            predicted_value = self._softmax(np.dot(X, self.w))
            predicted_value = np.argmax(predicted_value, axis = 1)

            return predicted_value 


    def _compute_gradients(self, X: np.ndarray, y: np.ndarray):
        if self.model_type == "linear":
            # TODO: 3%
            count_x = X.shape[0]
            for i in range(self.iterations):
                gradient_w = 0
                gradient_b = 0
                for j in range(count_x):
                    predicted_value = np.dot(X[j], self.w) + self.b
                    gradient_w += -X[j] * (y[j]-predicted_value)
                    gradient_b += -(y[j]-predicted_value)

                gradient_w *= 2/count_x
                self.w -= self.learning_rate * gradient_w

                gradient_b *= 2/count_x
                self.b -= self.learning_rate * gradient_b
                
        elif self.model_type == "logistic":
            # TODO: 3%
            count_x = X.shape[0]

            for i in range(self.iterations):
                z = np.matmul(X, self.w)
                p = self._softmax(z)
                gradient_w = 1/count_x * np.matmul(X.T, (p-y))
                self.w -= self.learning_rate * gradient_w


    def _softmax(self, z: np.ndarray) -> np.ndarray :
        #print(z)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree: 
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        mask = X[:, feature] <= threshold
        left_X, right_X, left_y, right_y = X[mask], X[~mask], y[mask], y[~mask]
        
        left_child = self._build_tree(left_X, left_y, depth+1)
        right_child = self._build_tree(right_X, right_y, depth+1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            uniq, count = np.unique(y, return_counts=True)
            index = np.argmax(count)
            return(uniq[index])
        else:
            # TODO: 1%
            return(np.mean(y))

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        #for classification
        right_gini = 0
        left_gini = 0

        #left
        classes = np.unique(left_y)
        len_left_y= len(left_y)
        for i in classes:
            p = (len(np.array([j for j in left_y if j == i]))) / len_left_y
            left_gini += p**2

        left_gini = 1-left_gini

        #right
        classes = np.unique(right_y)
        len_right_y= len(right_y)
        for i in classes: 
            p = (len(np.array([j for j in right_y if j == i]))) / len_right_y
            right_gini += p**2

        right_gini = 1-right_gini

        total_len = len_left_y + len_right_y
        ans = (len_left_y/total_len) * left_gini + (len_right_y/total_len) * right_gini

        return ans


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        #for regression
        
        len_left_y, len_right_y = len(left_y), len(right_y)
        total_len = len_left_y + len_right_y

        ans = (len_left_y/total_len) * np.var(left_y) + (len_right_y/total_len) * np.var(right_y)
        return ans



    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.forest = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_features = X.shape[1]
        self.select_features = int(round(X.shape[1] ** (0.5),0))
        count_X = X.shape[0]
        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(count_X, count_X)
            new_X = X[bootstrap_indices]
            new_y =  y[bootstrap_indices]
            tree = DecisionTree(self.max_depth, self.model_type)
            tree.fit(new_X, new_y)
            self.forest.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        temp = True
        for i in self.forest:
            if temp == True:
                temp = False
                all_answers = np.array(i.predict(X))

            else:
                all_answers = np.vstack((all_answers, i.predict(X)))

        all_answers = all_answers.T

        count_X = X.shape[0]
        answer = np.zeros(count_X)

        if self.model_type == "classifier":
            for i in range(count_X):
                uniq, count = np.unique(all_answers[i], return_counts=True)
                index = np.argmax(count)
                answer[i] = uniq[index]
        else:
            for i in range(count_X):
                answer[i] = np.mean(all_answers[i])


        return answer
 


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%   

    y_count = len(y_pred)
    correct = 0
    for i in range(y_count):
        if y_pred[i] == y_true[i]:
            correct += 1
    
    return (correct/y_count)

def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    mse = np.mean((y_true - y_pred) ** 2)
    return mse




# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)
    
    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(n_estimators= 100, max_depth = 5, model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    #Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(n_estimators= 100, max_depth = 20, model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
