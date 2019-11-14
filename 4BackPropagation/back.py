import numpy as np # numpy is commonly used to process number array

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) # Features ( Hrs Slept, Hrs Studied) 
y = np.array(([92], [86], [89]), dtype=float)	# Labels(Marks obtained)

X = X/np.amax(X,axis=0) # Normalize 
y = y/100

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_grad(x):
    return x * (1 - x)

# Variable initialization
epoch=1000	#Setting training iterations
eta =0.2		#Setting learning rate (eta)
input_neurons = 2	#number of features in data set 
hidden_neurons = 3	#number of hidden layers neurons
output_neurons = 1	#number of neurons at output layer

# Weight and bias - Random initialization
wh=np.random.uniform(size=(input_neurons,hidden_neurons))	# 2x3 
bh=np.random.uniform(size=(1,hidden_neurons))	# 1x3 
wout=np.random.uniform(size=(hidden_neurons,output_neurons)) # 1x1 
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
#Forward Propogation
    h_ip=np.dot(X,wh) + bh	# Dot product + bias 
    h_act = sigmoid(h_ip)	# Activation function 
    o_ip=np.dot(h_act,wout) + bout
    output = sigmoid(o_ip)

#Backpropagation
    # Error at Output layer
    Eo = y-output	# Error at o/p 
    outgrad = sigmoid_grad(output)
    d_output = Eo* outgrad	# Errj=Oj(1-Oj)(Tj-Oj)

    # Error at Hidden later
    Eh = d_output.dot(wout.T)	# .T means transpose
    hiddengrad = sigmoid_grad(h_act)	# How much hidden layer wts contributed to error 
    d_hidden = Eh * hiddengrad
    wout += h_act.T.dot(d_output) *eta	# Dotproduct of nextlayererror and currentlayerop 
    wh += X.T.dot(d_hidden) *eta

print("Normalized Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)