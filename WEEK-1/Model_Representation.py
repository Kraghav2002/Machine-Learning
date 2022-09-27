import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([1.0 , 2.0])
y_train = np.array([300.0 , 500.0])
print(f"x_train is {x_train}")
print(f"y_train is {y_train}")
m = len(x_train)
print(f"NUMBER OF TRAINING EXAMPLES ARE {m}")
​
plt.scatter(x_train , y_train)
plt.title("HOUSING PRICES")
plt.ylabel("PRICE OF HOUSES")
plt.xlabel("SIZE OF HOUSES")
plt.show()
​
w = 200
b = 100
​
def compute_model_output(x,w,b):
    m = len(x_train)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb
​
tmp_f_wb = compute_model_output(x_train,w,b)
plt.plot(x_train , tmp_f_wb)
plt.scatter(x_train , y_train)
plt.title("HOUSING PRICES")
plt.ylabel("PRICE OF HOUSES")
plt.xlabel("SIZE OF HOUSES")
plt.show()
​
