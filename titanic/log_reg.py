import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
titanic = pd.read_csv('titanic_cld.csv')
titanic_new = pd.read_csv('test_cld.csv')
titanic_new = titanic_new.iloc[:, 1:]
titanic_y = titanic.iloc[:, 1]
titanic_x = titanic.iloc[:, 2:]

# Create the scaler object and fit it on the data
scaler = StandardScaler()
scaler.fit(titanic_x)

# Apply the scaler to the data
titanic_x_scaled = scaler.transform(titanic_x)
titanic_new_scaled = scaler.transform(titanic_new)


log_reg = LogisticRegression()
log_reg.fit(titanic_x_scaled, titanic_y)

#print(titanic_x_scaled)
#print(titanic_y)

print(log_reg.score(titanic_x_scaled, titanic_y))

# print results
y_proba = log_reg.predict(titanic_new_scaled)
#y_proba = pd.DataFrame(y_proba)
#print(y_proba)
#y_proba.to_csv('result.csv')
print(y_proba.reshape(-1, 1))
# visiualize probe
train_proba = log_reg.predict_proba(titanic_new_scaled)
x = np.linspace(1, 418, num=418).reshape(-1, 1)
print(x)
plt.scatter(x, y_proba.reshape(-1, 1))
plt.xlabel('Predicted probability of survival')
plt.ylabel('Actual outcome (0 = died, 1 = survived)')
plt.show()