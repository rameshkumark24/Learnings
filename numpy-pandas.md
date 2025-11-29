# ✅ ONE-PAGE NUMPY CHEAT SHEET  
(Fresher + Company-Level Interview Ready)

---

## ✅ 1. Basic Import

```python
import numpy as np
````

---

## ✅ 2. Creating Arrays

```python
np.array([1, 2, 3])
np.zeros((3, 3))
np.ones((2, 2))
np.arange(0, 10, 2)
np.linspace(0, 1, 5)
np.eye(3)          # Identity Matrix
```

---

## ✅ 3. Array Operations

```python
a + b
a - b
a * b
a / b

np.dot(a, b)       # Matrix Multiplication
np.sqrt(a)
np.mean(a)
np.sum(a)
np.std(a)
```

---

## ✅ 4. Indexing & Slicing

```python
a[0]
a[-1]
a[1:4]

a[:, 0]           # First column
a[0, :]           # First row
```

---

## ✅ 5. Reshaping & Transpose

```python
a.reshape(3, 4)
a.flatten()
a.T               # Transpose
```

---

## ✅ 6. Stacking Arrays

```python
np.hstack([a, b])
np.vstack([a, b])
```

---

## ✅ 7. Useful NumPy Functions

```python
np.unique(a)
np.argmax(a)
np.argmin(a)
np.where(a > 10)
```

---

## ✅ 8. Random Numbers (Very Important for ML)

```python
np.random.rand(3,3)
np.random.randn(3,3)
np.random.randint(0, 100, size=10)
```

---

## ✅ 9. Broadcasting (Interview Favorite)

```python
a = np.array([1, 2, 3])
a + 10
```

➡️ Automatically adds `10` to all elements

---

## ✅ 10. Boolean Masking

```python
a[a > 5]
```

---

# ⚡ NUMPY INTERVIEW KEY POINTS

✅ Faster than Python lists
✅ Uses C backend → High performance
✅ Supports vectorization
✅ Foundation for Pandas, ML, Deep Learning

---

---

# ✅ ONE-PAGE PANDAS CHEAT SHEET

(Fresher + Company-Level Interview Ready)

---

## ✅ 1. Basic Import

```python
import pandas as pd
```

---

## ✅ 2. Read & Write Data

```python
df = pd.read_csv("data.csv")
df.to_csv("output.csv", index=False)
```

---

## ✅ 3. Understanding the Data

```python
df.head()
df.tail()
df.info()
df.describe()
df.columns
df.shape
```

---

## ✅ 4. Selecting Columns & Rows

```python
df["col"]
df[["col1", "col2"]]

df.iloc[0]          # By index
df.loc[3]           # By label
df.loc[:, "col"]    # Full column
```

---

## ✅ 5. Filtering Data

```python
df[df["age"] > 30]

df[(df["age"] > 30) & (df["salary"] > 50000)]

df[df["gender"] == "Male"]
```

---

## ✅ 6. Handling Missing Values

```python
df.isnull().sum()
df.dropna()
df.fillna(0)
df.fillna(df["age"].mean())
```

---

## ✅ 7. Add & Remove Columns

```python
df["new_col"] = df["salary"] * 12
df.drop("col", axis=1, inplace=True)
```

---

## ✅ 8. GroupBy (Most Important Topic)

```python
df.groupby("department")["salary"].mean()

df.groupby("gender").agg({
    "age": "mean",
    "salary": "sum"
})
```

---

## ✅ 9. Sorting Data

```python
df.sort_values("age")

df.sort_values(["age", "salary"], ascending=[True, False])
```

---

## ✅ 10. Merge & Concat

```python
pd.merge(df1, df2, on="id")

pd.concat([df1, df2])
```

---

## ✅ 11. Apply & Lambda

```python
df["age_group"] = df["age"].apply(
    lambda x: "Adult" if x >= 18 else "Child"
)
```

---

## ✅ 12. Correlation

```python
df.corr()
```

---

## ✅ 13. Value Counts & Unique (Interview Favorite)

```python
df["gender"].value_counts()
df["department"].unique()
df["department"].nunique()
```

---

## ✅ 14. Data Type Conversion

```python
df["age"] = df["age"].astype(int)
```

---

## ✅ 15. Pivot Table

```python
df.pivot_table(
    values="salary",
    index="department",
    aggfunc="mean"
)
```

---

## ✅ 16. Binning Data

```python
pd.cut(df["age"], bins=3)
pd.qcut(df["salary"], q=4)
```

---

# ⚡ 15 MOST ASKED PANDAS INTERVIEW FUNCTIONS

groupby()
agg()
apply()
merge() vs concat()
loc vs iloc
value_counts()
isnull()
drop_duplicates()
fillna()
unique()
nunique()
map() vs apply()
astype()
pivot_table()
cut() / qcut()

---

# ✅ FINAL INTERVIEW TIPS

✅ NumPy → Speed + Math backend
✅ Pandas → Data Cleaning + Analysis
✅ GroupBy + Merge + Apply = 80% Interview Cleared
✅ NumPy + Pandas mastery = Strong ML Foundation
