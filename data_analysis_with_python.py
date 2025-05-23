############################################
# Data Analysis with Python
############################################
## - NumPy
## - Pandas
## - Data Visualization ( Matplotlib & Seaborn )
##- Advanced Functional Exploratory Data Analysis

############################################
## NumPy
############################################
### Why NumPy
    #### - NumPy is commonly used within data science in order to work through numerical analyses and functions, such as creating and working with arrays, returning descriptive statistics, and a variety of machine learning models and mathematical formulas.
import numpy as np
from numpy.conftest import dtype

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

### Creating NumPy Arrays
np.array([1, 2, 3, 4, 5])

type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype = int)

np.ones(5)

np.arange(10)

np.arange(2, 10, dtype = float)

np.arange(2, 3, 0.1)

np.random.randint(0 , 10, size = 10)

np.random.normal(10, 4, (3, 4))

np.eye(3)

np.eye(3, 5)

np.diag([1, 2, 3])

np.diag([1, 2, 3], 1)

a = np.array([[1, 2], [3, 4]])
np.diag(a)

a = np.linspace(1, 4, 4)
np.vander(a, 2)
np.vander(a, 4)

np.indices((3,3))

### Attributes of NumPy Arrays
#### ndim : returns an integer that tells us how many dimensions the array have
#### shape : the number of elements in each dimension
#### size : a fixed size list of elements
#### dtype : informs us about the layout of the array
#### itemsize : determines size (in bytes) of each element in the array
#### data : returns the memory addresses of the data

a = np.random.randint(10, size = 5, dtype = np.int64)
a.ndim
a.shape
a.size
a.dtype
a.itemsize
a.data

a = np.random.randint(1, 10, size = (4, 4))
a.ndim
a.shape
a.size
a.dtype
a.itemsize
a.data

### Reshaping
a = np.arange(6).reshape((3, 2))

np.reshape(a, (2, 3)) # C-like index ordering

np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape

np.reshape(a, (2, 3), order='F') # Fortran-like index ordering

np.reshape(np.ravel(a, order='F'), (2, 3), order='F')


np.random.randint(1, 10, size = 9)
np.random.randint(1, 10, size = 9).reshape(3,3)

ar = np.random.randint(1, 10, size = 9)
ar.reshape(3, 3)

ar.reshape(3, 4)
### Index Selection & Slicing
a = np.random.randint(10, size = 10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(1, 10, size = (3, 5))

m[0, 0]
m[2, 3] = 999

m[2, 3] = 2.9

m[:, 0]

m[1, :]
m[:2, :3]
### Fancy Index
a = np.arange(0, 30, 3)

a[1]
a[4]

catch = [1, 2, 3]
a[catch]

### Conditions on NumPy
a = np.arange(1, 6)

#### classic way
ab = []
for i in a:
    if i < 3:
        ab.append(a[i])

#### with numpy
a < 3
a[a < 3]
a[a >= 3]
### Mathematical Operations

a / 5

a * 5 / 10

a - 1

np.add(a, 1)
np.mean(a)
np.sum(a)
np.min(a)
np.max(a)
np.var(a)

a = np.subtract(a, 1)

#### Solving equation with two unknowns with numpy
# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 2]])
b = np.array([12, 10])

np.linalg.solve(a, b)