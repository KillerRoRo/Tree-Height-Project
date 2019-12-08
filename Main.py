import numpy as np
import matplotlib.pyplot as plt
X = np.array(([1,0,0],[1,1,1.4],[1,1,3.4],[1,1,9.1],[1,1,1.6],[1,1,10.5], [1,1,2.5],[1,0,4.1],[1,0,1.6], [1,0,2],[1,0,5.5]),dtype = float)
Y = np.array((27,21,25,1,3,24,17,18,6,17,20),dtype = float)
X = X/np.amax(X, axis=0) # maximum of X array
Y = Y/27
X3 = (X-X.mean()) / X.std()
Y4 = (Y-Y.mean()) / Y.std()
alpha = 0.11053
iters = 148
X2 = np.matrix(X3)
y2 = np.matrix(Y4)
theta2 = np.matrix(np.array([0,0,0]))

class Neural_Network(object):

    def computeCost(self,X, y, theta):
        inner = np.power(((X * theta.T) - y), 2)
        return np.sum(inner) / (2 * len(X))
    def gradientDescent(self,X,Y,theta,alpha,iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.ravel().shape[1])
        cost = np.zeros(iters)

        for i in range(iters):
            error = (X*theta.T)-Y


            for j in range(parameters):
                h = np.multiply(error,X[:,j])
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(h))
            theta = temp
            cost[i] = abs(self.computeCost(X, Y, temp))
        return theta,cost





def main():
    NN = Neural_Network()
    g2, cost2 = NN.gradientDescent(X2, y2, theta2, alpha, iters)
    print("THis is cost")
    print(cost2)

    plt.plot(np.arange(iters), cost2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs. Training Epoch')
    plt.show()




if __name__ == "__main__": main()






   
    


