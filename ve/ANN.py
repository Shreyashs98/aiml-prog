import numpy as np
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([90],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100
def sigmoid(X):
    return 1/(1+np.exp(-X))
def derivatives_sigmoid(X):
    return X*(1-X)
epoch=1000
learning_rate=0.6
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wo=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bo=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):
    net_h=np.dot(X,wh)+bh
    sigma_h=sigmoid(net_h)
    net_o=np.dot(sigma_h,wo)+bo
    output=sigmoid(net_o)
    deltaK=(y-output)*derivatives_sigmoid(output)
    deltaH=deltaK.dot(wo.T)*derivatives_sigmoid(sigma_h)
    wo=wo+sigma_h.T.dot(deltaK)*learning_rate
    wh=wh+X.T.dot(deltaH)*learning_rate
print(f"Input:\n{X}")
print(f"Actual Output:\n{y}")
print(f"Predicted Output:\n{output}")