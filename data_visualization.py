########################
# Data Visualization
########################
### BI tools that communicate with the database are more suitable for data visualization.
### Categorical variable    : bar chart - pie chart
### Numeric variable        : histogram, box

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

## Matplotlib
### Categorical variable
df["sex"].value_counts().plot(kind = 'bar')
plt.show()

### Numeric variable
plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

### Features
#### plot
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

##### marker
y = np.array([13, 28, 11, 100])

plt.plot(y, marker = 'H')
plt.show()

##### linestyle
plt.plot(y, linestyle = "dashdot")
plt.show()

#### Multiple lines
plt.plot(x)
plt.plot(y)
plt.show()

#### label
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)
plt.show()

##### Main title
plt.plot(x, y)
plt.title("This is main title")
##### x - y naming
plt.xlabel("X label")
plt.ylabel("Y label")

plt.grid()
plt.show()

#### Subplots
##### plot1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)
##### plot2
x = np.array([8, 8, 9, 9, 10, 10, 11, 11, 12, 12])
y = np.array([24, 25, 26, 27, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)

## Seaborn

df = sns.load_dataset("tips")
df.head()

### Categorical variable
sns.countplot(x = df["sex"], data = df)
plt.show()

### Numeric variable
sns.boxplot(x = df["total_bill"], data = df)
plt.show()

df["total_bill"].hist()
