# importing the required libraries
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile, chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, \
    BaggingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold, learning_curve, \
    train_test_split, KFold
# from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sb
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


# ----------------------------------------------------------------------------------------------------------------------

# PART - 1: Establish a baseline
# ----------------------------------------------------------------------------------------------------------------------
# function to remove null values within the data
def remove_null(data_ref):
    # modifying the ? to NaN values in the data
    data_ref.replace(" ?", np.nan, inplace=True)
    print("The count of null values present in the columns:")
    print(data_ref.isnull().sum())
    # now finding the most frequent values for the columns which are having null values
    wc = data_ref['workclass'].value_counts().idxmax()
    oc = data_ref['occupation'].value_counts().idxmax()
    nc = data_ref['native-country'].value_counts().idxmax()
    # now replacing those null values with these frequent values w.r.t their columns
    data_ref["workclass"].replace(np.nan, wc, inplace=True)
    data_ref["occupation"].replace(np.nan, oc, inplace=True)
    data_ref["native-country"].replace(np.nan, nc, inplace=True)
    print("The count after removing the null values:")
    print(data_ref.isnull().sum())
    return data_ref


# function to handle outliers
def outliers(data_ref):
    print("Describing the skewness in data:")
    print(data_ref.skew())
    # fetching the columns with numerical data
    num_col = list(data_ref.select_dtypes(include=["int64"]).columns)
    print("Columns having integer values")
    print(num_col)
    l_limits = []
    u_limits = []
    QR_values = []
    # Detecting outliers using the Inter Quantile Range(IQR)
    for i in range(len(num_col)):
        q1 = data_ref[num_col[i]].quantile(0.25)
        q3 = data_ref[num_col[i]].quantile(0.75)
        QR = q3 - q1
        QR_values.append(QR)
        l_limit = q1 - (1.5 * QR)
        l_limits.append(l_limit)
        u_limit = q3 + 1.5 * QR
        u_limits.append(u_limit)
    QR = pd.DataFrame({"numeric_columns": num_col, "lower_limits": l_limits,
                       "upper_limits": u_limits, "IQR_values": QR_values})
    print("Describing the integer columns with IQR limits:")
    print(QR)
    print("outlier number for hours-per-week : {}".format(
        data_ref[(data_ref["hours-per-week"] < (l_limits[4])) | (data_ref["hours-per-week"] > (u_limits[4]))].shape[0]))
    print("Final Weight Outlier Number :{}".format(
        data_ref[(data_ref["fnlwgt"] < (l_limits[1])) | (data_ref["fnlwgt"] > (u_limits[1]))].shape[0]))
    # removing the outliers from final weight column
    data_ref.drop(data_ref[data_ref["fnlwgt"] > 900000].index, inplace=True)
    print("Describing the data of the integer columns:")
    print(round(data_ref[num_col].describe(), 2))
    return data_ref


# function for handling categorical data
def handling_categorical(data_ref):
    print("Data before handling the categories:")
    print(data_ref.head())
    # dropping the unwanted columns
    data_ref.drop(['marital-status'], axis=1, inplace=True)
    data_ref.drop(['education'], axis=1, inplace=True)
    # handling the categorical data with the help of LableEncoder
    labels = ['workclass', 'occupation', 'native-country', 'sex', 'income', 'race', 'relationship']
    le = LabelEncoder()
    for l in labels:
        data_ref[l] = le.fit_transform(data_ref[l])
    print("After handling the categorical data:")
    print(data_ref.head())
    return data_ref


# function for scaling the data and handling the imbalance
def scaling_imbalance(data_ref):
    # scaling the data with standard scalar
    X = StandardScaler().fit_transform(data_ref.loc[:, data_ref.columns != 'income'])
    Y = data_ref['income']
    # Handling imbalance data using SMOTE
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    plot_sb = sb.countplot(Y_train, label='Total')
    plt.show()
    G_50, L_50 = Y_train.value_counts()
    print('<=50K: ', L_50)
    print('>50K : ', G_50)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    sm = SMOTE(random_state=0)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)
    plot_sb = sb.countplot(Y_train, label='Total')
    plt.show()
    G_50, L_50 = Y_train.value_counts()
    print('<=50K: ', L_50)
    print('>50K : ', G_50)
    return X_train, X_test, Y_train, Y_test


# 1. pre_processing the Data
def pre_process_data(data_ref):
    # dealing with the missing values
    data_ref = remove_null(data_ref)
    # dealing with outliers
    data_ref = outliers(data_ref)
    # Handling Categorical Data
    data_ref = handling_categorical(data_ref)
    # Scaling data and handling imbalance
    X_train, X_test, Y_train, Y_test = scaling_imbalance(data_ref)
    return X_train, X_test, Y_train, Y_test


# 2. building basic models
def build_models(X_train, X_test, Y_train, Y_test):
    # taking some classifiers
    classifiers = [LogisticRegression(solver='newton-cg'),
                   KNeighborsClassifier(n_neighbors=17),
                   LinearDiscriminantAnalysis(),
                   GaussianNB(),
                   RidgeClassifier(), GradientBoostingClassifier(),
                   SVC(), RandomForestClassifier()]
    # predicting the F1 score for each classifier
    for classifier in classifiers:
        print(type(classifier).__name__)
        pipe = Pipeline(steps=[('classifier', classifier)])
        pipe.fit(X_train, Y_train)
        Y_pred = pipe.predict(X_test)
        score = pipe.score(X_train, Y_train)
        print("F1 score: {:}".format(f1_score(Y_test, Y_pred)))

# 3. Hyper parameter tuning
def hyper_parameter_tuning(X_train, X_test, Y_train, Y_test):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    objrandom_grid = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      }
    # Random Forest Classifier with Hyper parameter tunning
    obj_rf = RandomForestClassifier()
    obj_rf = RandomizedSearchCV(estimator=obj_rf, scoring="f1", param_distributions=objrandom_grid, n_iter=10, cv=3,
                               verbose=2, random_state=42, n_jobs=-1, )
    obj_rf.fit(X_train, Y_train)
    print("Result after applying hyper parameter tuning for Random Forest Classifier:")
    print("Best Estimator:", obj_rf.best_estimator_)
    print("Best Score", obj_rf.best_score_)
    # GradientBoostingClassifier with hyper parameter tunning
    obj_gbc = GradientBoostingClassifier()
    obj_gbc = RandomizedSearchCV(estimator=obj_gbc, scoring="f1", param_distributions=objrandom_grid, n_iter=10,
                                      cv=3, verbose=2, random_state=42, n_jobs=-1, )
    obj_gbc.fit(X_train, Y_train)
    print("Result after applying hyper parameter tuning for Gradient Boosting Classifier:")
    print("Best Estimator:", obj_gbc.best_estimator_)
    print("Best Score", obj_gbc.best_score_)
    # Logistic Regression with hyper parameter tunning
    C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    sepal_acc_table = pd.DataFrame(columns=['C_parameter', 'f1 score'])
    sepal_acc_table['C_parameter'] = C_param_range
    j = 0
    for i in C_param_range:
        # Apply logistic regression model to training data
        lr = LogisticRegression(penalty='l2', C=i, random_state=0)
        lr.fit(X_train, Y_train)
        # Predict using model
        y_pred = lr.predict(X_test)
        # Saving accuracy score in table
        sepal_acc_table.iloc[j, 1] = f1_score(Y_test, y_pred)
        j += 1
    print("Result after applying hyper parameter tuning for Logistic Regression:")
    print(max(sepal_acc_table['f1 score']))

# ----------------------------------------------------------------------------------------------------------------------

# PART - 2: Basic Experimentation
# ----------------------------------------------------------------------------------------------------------------------
def feature_selection(X_train, X_test, Y_train, Y_test):
    a = SelectPercentile(chi2, percentile=10).fit(X_train, Y_train)
    X_train = a.transform(X_train)
    X_test = a.transform(X_test)
    return X_train, X_test, Y_train, Y_test


def basic_experimentation(data_ref):
    # dealing with the missing values
    data_ref = remove_null(data_ref)
    # dealing with outliers
    data_ref = outliers(data_ref)
    # handling categorical data
    print("Data before handling the categories:")
    print(data_ref.head())
    # dropping the unwanted columns
    data_ref.drop(['marital-status'], axis=1, inplace=True)
    data_ref.drop(['education'], axis=1, inplace=True)
    # handling the categorical data with the help of LableEncoder
    labels = ['sex', 'income', 'race', 'relationship']
    le = LabelEncoder()
    for l in labels:
        data_ref[l] = le.fit_transform(data_ref[l])
    print("After handling the categorical data:")
    print(data_ref.head())
    # OneHot Encoder Implementation
    data_ref = pd.get_dummies(data_ref, columns=['workclass'], drop_first=True)
    data_ref = pd.get_dummies(data_ref, columns=['occupation'], drop_first=True)
    data_ref = pd.get_dummies(data_ref, columns=['native-country'], drop_first=True)
    # implementing MinMax scalar
    X = MinMaxScaler().fit_transform(data_ref.loc[:, data_ref.columns != 'income'])
    Y = data_ref['income']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    sm = SMOTE(random_state=0)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)
    X_train, X_test, Y_train, Y_test = feature_selection(X_train, X_test, Y_train, Y_test)
    build_models(X_train, X_test, Y_train, Y_test)

# ----------------------------------------------------------------------------------------------------------------------
# PART - 3: Research
# ----------------------------------------------------------------------------------------------------------------------
def logistic_regression(X_train, X_test, Y_train, Y_test):
    classifiers = [
        LogisticRegression()
    ]
    # performing logistic regression
    for classifier in classifiers:
        print(type(classifier).__name__)
        pipe = Pipeline(steps=[('classifier', classifier)])
        pipe.fit(X_train, Y_train)
        Y_pred = pipe.predict(X_test)
        score = pipe.score(X_train, Y_train)
        print("F1 score: {:}".format(f1_score(Y_test, Y_pred)))


def check_near_miss1():
    data_ref = pd.read_csv('adult.csv', delimiter=",",
                           names=["age", "workclass", "fnlwgt", "education", "education-num",
                                  "marital-status", "occupation", "relationship", "race",
                                  "sex",
                                  "capital-gain", "capital-loss", "hours-per-week",
                                  "native-country", "income"])
    print("Implementing Near Miss 1:")
    # dealing with the missing values
    data_ref = remove_null(data_ref)
    # dealing with outliers
    data_ref = outliers(data_ref)
    # Handling Categorical Data
    data_ref = handling_categorical(data_ref)
    # Scaling data and handling imbalance
    X_train, X_test, Y_train, Y_test = scaling_imbalance(data_ref)
    undersample1 = NearMiss(version=1, n_neighbors=3)
    # transform the dataset
    X_train, Y_train = undersample1.fit_resample(X_train, Y_train)
    plot_sb = sb.countplot(Y_train, label='Total')
    plt.show()
    G_50, L_50 = Y_train.value_counts()
    print('<=50K: ', L_50)
    print('>50K : ', G_50)
    logistic_regression(X_train, X_test, Y_train, Y_test)


def check_near_miss2():
    data_ref = pd.read_csv('adult.csv', delimiter=",",
                           names=["age", "workclass", "fnlwgt", "education", "education-num",
                                  "marital-status", "occupation", "relationship", "race",
                                  "sex",
                                  "capital-gain", "capital-loss", "hours-per-week",
                                  "native-country", "income"])
    print("Implementing Near Miss 2:")
    # dealing with the missing values
    data_ref = remove_null(data_ref)
    # dealing with outliers
    data_ref = outliers(data_ref)
    # Handling Categorical Data
    data_ref = handling_categorical(data_ref)
    # Scaling data and handling imbalance
    X_train, X_test, Y_train, Y_test = scaling_imbalance(data_ref)
    undersample2 = NearMiss(version=2, n_neighbors=3)
    # transform the dataset
    X_train, Y_train = undersample2.fit_resample(X_train, Y_train)
    plot_sb = sb.countplot(Y_train, label='Total')
    plt.show()
    G_50, L_50 = Y_train.value_counts()
    print('<=50K: ', L_50)
    print('>50K : ', G_50)
    logistic_regression(X_train, X_test, Y_train, Y_test)


def check_near_miss3():
    data_ref = pd.read_csv('adult.csv', delimiter=",", names=["age", "workclass", "fnlwgt", "education", "education-num",
                                                           "marital-status", "occupation", "relationship", "race",
                                                           "sex",
                                                           "capital-gain", "capital-loss", "hours-per-week",
                                                           "native-country", "income"])
    print("Implementing Near Miss 3:")
    # dealing with the missing values
    data_ref = remove_null(data_ref)
    # dealing with outliers
    data_ref = outliers(data_ref)
    # Handling Categorical Data
    data_ref = handling_categorical(data_ref)
    # Scaling data and handling imbalance
    X_train, X_test, Y_train, Y_test = scaling_imbalance(data_ref)
    undersample3 = NearMiss(version=3, n_neighbors=3)
    # transform the dataset
    X_train, Y_train = undersample3.fit_resample(X_train, Y_train)
    plot_sb = sb.countplot(Y_train, label='Total')
    plt.show()
    G_50, L_50 = Y_train.value_counts()
    print('<=50K: ', L_50)
    print('>50K : ', G_50)
    logistic_regression(X_train, X_test, Y_train, Y_test)


def research():
    check_near_miss1()
    check_near_miss2()
    check_near_miss3()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # reading the dataset
    data1 = pd.read_csv('adult.csv', delimiter=",", names=["age", "workclass", "fnlwgt", "education", "education-num",
                                                          "marital-status", "occupation", "relationship", "race", "sex",
                                                          "capital-gain", "capital-loss", "hours-per-week",
                                                          "native-country", "income"])
    # Data pre-processing
    X_train, X_test, Y_train, Y_test = pre_process_data(data1)
    # building models
    build_models(X_train, X_test, Y_train, Y_test)
    # hyper parameter tuning
    hyper_parameter_tuning(X_train, X_test, Y_train, Y_test)
    data2 = pd.read_csv('adult.csv', delimiter=",", names=["age", "workclass", "fnlwgt", "education", "education-num",
                                                          "marital-status", "occupation", "relationship", "race", "sex",
                                                          "capital-gain", "capital-loss", "hours-per-week",
                                                          "native-country", "income"])
    # basic experimentation
    print("Basic Experimentation:")
    basic_experimentation(data2)
    # research
    print("Research")
    research()
