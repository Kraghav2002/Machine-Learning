import copy , math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 2)

#TRAINING SET

x_train = np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train = np.array([460,232,178])

print(f"x_train.shape : {x_train.shape}")
print(f"y_train.shape : {y_train.shape}")
print(f"x_train : {x_train}")
print(f"y_train : {y_train}")

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict(x,w,b):
    p = np.dot(x,w) + b #dot product of x and w.
    return p

x_vec = x_train[0,:]
print(x_vec)
f_wb = predict(x_vec , w_init , b_init)
print(f_wb)

#COST COMPUTING FUNCTION:
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(x[i] , w) + b
        cost = cost + ((f_wb_i - y[i])**2)
    cost = cost / (2*m)
    return cost

cost = compute_cost(x_train , y_train , w_init , b_init)
print(f"THE CALCULATED COST IS {cost}")

def compute_gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n))
    dj_db = 0
    for i in range(m):
        err = np.dot(x[i] , w) + y[i]
        for j in range(n):
            dj_dw = dj_dw + err * x[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_dw , dj_db

tmp_dj_dw , tmp_dj_db = compute_gradient(x_train , y_train , w_init , b_init)
print(f"tmp_dj_dw : {tmp_dj_dw}")
print(f"tmp_dj_db : {tmp_dj_db}")

def compute_gradient_descent(x,y,w_in,b_in,cost_function , gradient_function , alpha , num_iters):
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        
    
        dj_dw , dj_db = gradient_function(x, y , w , b)
    
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
        if i<100000:      
            j_history.append( cost_function(x, y, w, b))
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}")
            
    return w , b , j_history


initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 1000
alpha = 5.0e-7
w_final, b_final, j_hist = compute_gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
        
    

