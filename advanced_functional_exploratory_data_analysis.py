########################
# Advanced Functional EDA
########################
from statistics import quantiles

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fontTools.unicodedata import block

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

## General look
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head = 5):
    print("############# Shape #############")
    print(dataframe.shape)
    print("############# Types #############")
    print(dataframe.dtypes)
    print("############# Head #############")
    print(dataframe.head(head))
    print("############# Tail #############")
    print(dataframe.tail(head))
    print("############# NA #############")
    print(dataframe.isnull().values.any())
    print("############# Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df = sns.load_dataset("tips")
check_df(df)

## Analysis of Categorical Variables
df = sns.load_dataset("titanic")

df["sex"].value_counts()
df["sex"].unique()
df["sex"].nunique()

df.info()
categoric_variables = ["bool", "object", "category"]
cat_cols = [col for col in df.columns if str(df[col].dtypes) in categoric_variables]

numeric_variables = ["int", "float"]
numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in numeric_variables]

categoric_but_cardinality = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in categoric_variables]

cat_cols += numeric_but_categoric

cat_cols = [col for col in cat_cols if col not in categoric_but_cardinality]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

# Let's start the function
# Get the value count information of the columns of the given dataframe
# Let's calculate the ratios of the data in the columns
# !!!!! Do one thing principle so this is main function !!!!!
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

# Adding unnecessary properties to the function just for example
def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)

df.info()

for col in cat_cols:
    cat_summary(df, col, plot = True)


## Analysis of Numerical Variables
numeric_variables = ["int", "float"]
categoric_variables = ["bool", "object", "category"]

cat_cols = [col for col in df.columns if str(df[col].dtypes) in categoric_variables]
numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in numeric_variables]
categoric_but_cardinality = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in categoric_variables]
cat_cols += numeric_but_categoric
cat_cols = [col for col in cat_cols if col not in categoric_but_cardinality]

df.head()
df.info()
df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in numeric_variables]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot :
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

for col in num_cols:
    num_summary(df, col, plot = True)

## Capturing Variables and Generalizing Operations
df.head()
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def grab_col_names(dataframe, categoric_threshold = 10, cardinality_threshold = 20):
    """
    Gives the names of categorical, numeric and categorical but cardinal variables in the data set

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe on which variable names will be processed
    categoric_threshold : int, float
        Threshold value for numeric but categorical variables
    cardinality_threshold : int, float
        Threshold for categorical but cardinal variables

    Returns
    ---------
    cat_cols : list
        Categorical variable list
    num_cols : list
        Numerical variable list
    cat_but_car : list
        List of cardinal variables with categorical view
    """
    numeric_variables = ["int", "float"]
    categoric_variables = ["bool", "object", "category"]

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in categoric_variables]
    numeric_but_categoric = [col for col in df.columns if
                             df[col].nunique() < 10 and df[col].dtypes in numeric_variables]
    categoric_but_cardinality = [col for col in df.columns if
                                 df[col].nunique() > 20 and str(df[col].dtypes) in categoric_variables]
    cat_cols += numeric_but_categoric
    cat_cols = [col for col in cat_cols if col not in categoric_but_cardinality]

    num_cols = [col for col in df.columns if df[col].dtypes in numeric_variables]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"categoric_cols : {len(cat_cols)}")
    print(f"numeric_cols : {len(num_cols)}")
    print(f"numeric_but_categoric : {len(numeric_but_categoric)}")
    print(f"categoric_but_cardinality : {len(categoric_but_cardinality)}")

    return cat_cols, num_cols, categoric_but_cardinality


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)


for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)


## Analysis of Target Variables
### Analysis of Target Variable with Categorical variables
cat_summary(df,"survived")
df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN" : dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")

target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

### Analysis of Target Variable with Numerical variables
df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age" : "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col : "mean"}), end = "\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)


## Analysis of Correlation

df = pd.read_csv("datasets/breast-cancer.csv")
df = df.iloc[:, 2:]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int,float]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


### Deletion of highly correlated variables

cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)