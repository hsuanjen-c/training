import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

#data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

#model
"""b1 = np.ones((2,1))
b2 = np.ones((1,1))
W1 = np.ones((2,2))
W2 = np.ones((1,2))"""
W1 = np.random.randn(2, 2) * 0.01
W2 = np.random.randn(1, 2) * 0.01
b1 = np.random.randn(2, 1) * 0.01
b2 = np.random.randn(1, 1) * 0.01

def y_pred(Xi):
    return sigmoid(W2@sigmoid(W1@Xi+b1)+b2)

#loss function
def loss_function():
    loss = 0
    for Xi,yi in zip(X,y):
        Xi = Xi.reshape(2,1)
        output = sigmoid(W2@sigmoid(W1@Xi+b1)+b2)
        #print("output:",np.shape(W2@sigmoid(W1@Xi+b1)))
        error = float(yi) - float(output)
        loss += 0.25 * (error ** 2)
    return loss
loss = loss_function()
#gradient decent and training
def gradient_W2(W2):
    gradeint_i = np.zeros_like(W2)
    for Xi,yi in zip(X,y):
        Xi = Xi.reshape(2,1)
        A1 = sigmoid(W1@Xi+b1)
        A2 = sigmoid(W2@A1+b2)
        Z1 = W1@Xi+b1
        Z2 = W2@A1+b2
        error = float(yi)-float(A2)
        gradeint_i += sigmoid(Z2)*(1-sigmoid(Z2))*0.5*error*A1.T
        #gradeint_i += sigmoid(W2@sigmoid(W1@Xi+b1)+b2)*(1-sigmoid(W2*sigmoid(W1@Xi+b1)+b2))*0.5*(float(yi) - float(sigmoid(W2@sigmoid(W1@Xi+b1))))*(W1@Xi+b1)
    return gradeint_i/4
def gradient_W1(W1):
    gradeint_i = np.zeros_like(W1)
    for Xi,yi in zip(X,y):
        Xi = Xi.reshape(2,1)
        A1 = sigmoid(W1@Xi+b1)
        A2 = sigmoid(W2@A1+b2)
        Z1 = W1@Xi+b1
        Z2 = W2@A1+b2
        error = yi-A2
        dLdZ2 = sigmoid(Z2)*(1-Z2)*0.5*error #1,1
        dLdZ1 = W2.T@dLdZ2*sigmoid(Z1)*(1-sigmoid(Z1)) #w2 1*2 Z1:2*1
        gradeint_i += Xi@dLdZ1.T
    return gradeint_i/4
def gradient_b2(b2):
    gradeint_i = np.zeros_like(b2)
    for Xi,yi in zip(X,y):
        Xi = Xi.reshape(2,1)
        A1 = sigmoid(W1@Xi+b1)
        A2 = sigmoid(W2@A1+b2)
        Z1 = W1@Xi+b1
        Z2 = W2@A1+b2
        error = float(yi)-float(A2)
        gradeint_i+=sigmoid(Z2)*(1-sigmoid(Z2))*0.5*error
    return gradeint_i/4

    
learning_rate = 0.1
counter = 10000
while counter:
    counter-=1
    W2 = W2-learning_rate*gradient_W2(W2)
    W1 = W1-learning_rate*gradient_W1(W1)
    b2 = b2-learning_rate*gradient_b2(b2)
    loss = loss_function()
    if not (counter%1000):print(counter,":",loss)
for Xi,yi in zip(X,y):
    print("X:",Xi)
    print("y_pred:",y_pred(Xi))
print("W1:",W1)
print("W2:",W2)