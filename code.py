from google.colab import files
uploaded=files.upload() #choose file from your desktop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load Data
df = pd.read_csv('exams.csv')
df.head()

# Check for null values
df.isnull().sum()
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())
  
# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
df[col] = le.fit_transform(df[col])
df.head()
  
# Correlation
print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
#Create class
threshold = df['math score'].median()

df['Result'] = (df['math score'] > threshold).astype(int)
df.head()

#linear regression
X_slr = df[['reading score']]
y_slr = df['math score']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_slr, y_slr, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

print("RMSE:", mean_squared_error(y_test, y_pred)**0.5)
print("R2:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.show()

#multiple linear regression
X = df.drop(['math score'], axis=1)
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred)**0.5)
print("R2:", r2_score(y_test, y_pred))

sns.regplot(x=y_test, y=y_pred)
plt.show()

#decision tree
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(40,20))

plot_tree(
    dtr,
    feature_names=X.columns,
    filled=True,
    fontsize=8,
    max_depth=4
)

plt.title("Random Forest - Sample Tree")
plt.show()

y_pred = dtr.predict(X_test)

print("R2:", r2_score(y_test, y_pred))

import numpy as np
#random forest
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

y_pred = rfr.predict(X_test)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.figure(figsize=(40,20))

tree = rfr.estimators_[0]

plot_tree(
    tree,
    feature_names=X.columns,
    filled=True,
    fontsize=8,
    max_depth=4
)

plt.title("Random Forest - Sample Tree")
plt.show()
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

#logistic regression
X = df.drop(['Result'], axis=1)
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("confusion matrix:",confusion_matrix(y_test, y_pred))
#naive bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#svm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
#arima
df_arima = df.copy()
df_arima = df_arima.reset_index(drop=True)

df_arima['time'] = range(len(df_arima))
df_arima.set_index('time', inplace=True)
series = df_arima['math score']
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(series)
plt.title("Math Score Time Series")
plt.show()
series_diff = series.diff().dropna()

plt.plot(series_diff)
plt.title("Differenced Series")
plt.show()
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(2,1,2))
model_fit = model.fit()

print(model_fit.summary())
  pred = model_fit.predict(start=1, end=len(series)-1)
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(series[1:], pred))
print("RMSE:", rmse)
  plt.figure(figsize=(10,5))
plt.plot(series.values, label="Actual")
plt.plot(pred.values, label="Predicted")
plt.legend()
plt.show()
forecast = model_fit.forecast(steps=5)
print("Future values:", forecast)
