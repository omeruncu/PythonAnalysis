############################################
## Pandas
############################################
### Pandas Series
import pandas as pd

s = pd.Series([1, 22, 12, 1, 5])
type(s)

s.index
s.dtype
s.size
s.ndim
s.values # numpy array returns
s.head(3)
s.tail(3)
### Reading Data
df = pd.read_csv("datasets/Advertising.csv")
df.head()
### Quick look at Data
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()
### Selection
df[0:13]
df.drop(0, axis = 0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis = 0).head(10)

# df = df.drop(delete_indexes, axis = 0)
# df.drop(delete_indexes, axis = 0, inplace = True)

#### Convert variable to index

df["age"].head()
df.age.head()

df.index = df["age"]
df
df.drop("age", axis = 1).head()

df.drop("age", axis = 1, inplace = True)
df.head()

#### Convert index to variable
##### 1
df["age"]

df["age"] = df.index

df.drop("age", axis = 1, inplace = True)

##### 2

df.reset_index().head()

df = df.reset_index().head()

### Operations on variables
pd.set_option('display.max_columns', None)

"age" in df

df["age"].head() # returns pandas Series

df[["age"]].head() # returns pandas DataFrame

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]
df

col_names = ["age2", "age3"]
df.drop(col_names, axis = 1, inplace = True)
df

#### loc : label based selection
df.loc[:, df.columns.str.contains("age")].head()
df.loc[:, ~df.columns.str.contains("age")].head()

df.loc[0:3]
df.loc[0:3, "age"]
##### fancy selection
col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]
#### iloc : integer based selection
df.iloc[0:3]
df.iloc[0, 0]

df.iloc[0:3, 0:3]

#### Conditional Selection
df = sns.load_dataset("titanic")
df[df["age"] > 50].head()

df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age","class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age","class","sex"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male") & (df["embark_town"] == "Cherbourg"), ["age","class","sex","embark_town"]].head()
### Aggregation & Grouping
#### count
#### first
#### last
#### mean
#### median
#### min
#### max
#### std
#### var
#### sum

df = sns.load_dataset("titanic")

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age" : "mean"})
df.groupby("sex").agg({"age" : ["mean", "sum"]})

df.groupby("sex").agg({"age" : ["mean", "sum"], "survived" : "mean"})

df.groupby(["sex", "embark_town"]).agg({"age" : ["mean"], "survived" : "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age" : ["mean"],
                                                 "survived" : "mean",
                                                 "sex" : "count"})

df.groupby(["sex", "embark_town", "class"]).agg({"age" : ["mean"], "survived" : "mean"})

#### pivot table
df.pivot_table("survived", "sex", "embarked") # (default mean)

df.pivot_table("survived", "sex", "embarked", aggfunc = "std")

df.pivot_table("survived", "sex", ["embarked", "class"], aggfunc = "std")

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option("display.width", 500)
### Apply and Lambda
df = sns.load_dataset("titanic")
df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

(df["age"] / 10).head()
(df["age2"] / 10).head()
(df["age3"] / 10).head()

for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df[["age", "age2", "age3"]].apply(lambda x : x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : (x - x.mean()) / x.std()).head()

def standard_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()
### Joining
import numpy as np

m = np.random.randint(1,30, size = (5, 3))
df1 = pd.DataFrame(m, columns = ["var1", "var2", "var3"])
df2 = df1 + 99

#### Concat
pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index = True)

pd.concat([df1, df2], ignore_index = True, axis = 1)

#### Merge
df1 = pd.DataFrame({'employees' : ["john", 'dennis', 'mark', 'maria'],
                    'group' : ['accounting', 'engineering', 'engineering', 'hr']})
df2 = pd.DataFrame({'employees' : ["mark", 'john', 'dennis', 'maria'],
                    'start_date' : [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on = "employees")

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group' : ['accounting', 'engineering', 'hr'],
                    'manager' : ['Caner', 'Mustafa', 'Berkcan']})

df5 = pd.merge(df3, df4, on = 'group')