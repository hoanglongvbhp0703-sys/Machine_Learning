import numpy as np
from sklearn .linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_blobs

centers=[[-5,2],[-2,-2],[1,2],[5,-2]]
x,y =make_blobs(n_samples=2000,centers=centers,cluster_std=1.0,random_state=30)

x_train,x_tmp,y_train,y_tmp=train_test_split(x,y,test_size=0.4,random_state=1)
x_cv,x_test,y_cv,y_test=train_test_split(x_tmp,y_tmp,test_size=0.5,random_state=1)

train_mses=[]
cv_mses=[]
models=[]
polys=[]
scalers=[]

for degree in range(1,6):
    poly = PolynomialFeatures(degree,include_bias=False)
    x_train_mapped=poly.fit_transform(x_train)

    scaler = StandardScaler()
    x_train_scaled=scaler.fit_transform(x_train_mapped)

    model=LinearRegression()
    model.fit(x_train_scaled,y_train)

    yhat_train = model.predict(x_train_scaled)
    train_mses.append(mean_squared_error(yhat_train,y_train)/2)

    x_cv_mapped=poly.transform(x_cv)
    x_cv_scaled=scaler.transform(x_cv_mapped)
    yhat_cv=model.predict(x_cv_scaled)
    cv_mses.append(mean_squared_error(yhat_cv,y_cv)/2)

    models.append(model)
    polys.append(poly)
    scalers.append(scaler)

best_degree=np.argmin(cv_mses)+1
print("best degree:", best_degree)

x_test_mapped=polys[best_degree-1].transform(x_test)
x_test_scaled=scalers[best_degree-1].transform(x_test_mapped)
yhat_test=models[best_degree-1].predict(x_test_scaled)
test_mse=mean_squared_error(yhat_test,y_test)/2

print("Train MSE:", train_mses[best_degree-1])
print("CV MSE:", cv_mses[best_degree-1])
print("Test MSE:", test_mse)
