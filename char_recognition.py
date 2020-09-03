# Neural Networks
# Bonus Assignment
import numpy as np
from tqdm import tqdm
import  matplotlib.pyplot as plt
##############################      Main      #################################

def read_train_file(file="OCR_train.txt"):
    training_data_list = []
    train_file = open(file, "r")
    for line in train_file:
        line = list(line.replace(" ", ""))
        line = [int(x) * 2 - 1 for x in line if x != "\n"]
        training_data_list.extend([line[:]])
        features = np.array(training_data_list)[:, :64]
        labels   = np.array(training_data_list)[:, 64:]
        labels[labels == 0] = -1
        features[features==0] = -1
    return features, labels



def Step(x):
    return x>=0

def Sgn(x):
    return 2*(x>=0)-1

class Perceptorn(object):
    def __init__(self, n, dim, act_function = Sgn):
        '''
        a layer of Perceptorn neurons
        n: number of neurons
        dim: dim of feature vector
        act_function: activation function of Perceptorn Neuron
        '''
        self.n = n
        self.dim = dim
        self.W = np.zeros((self.dim, self.n))
        self.act = act_function

    def __call__(self, x):
        '''
        x is feature vectors with shape(batch, dim) in case of batch computation
            or with shape(dim, ) in case of single instance computation
        
        returns act(x.W) with shape(batch, n)
        '''
        return self.act(np.dot(x, self.W))

    def out(self, x): return self.__call__(x)

    def get_w(self): return self.W

    def classify(self, F):
        pred = self.out(F)
        return np.argmax(pred, axis = 1)

    def acc(self, F, L):
        return np.sum(np.sum((self.out(F) == L), axis=1) == L.shape[1])/L.shape[0]



class Optimizer(object):
    def __init__(self, model, lr=1, max_epoch = 20000, beta = 0.999):
        self.lr = lr
        self.epoch = 0
        self.max_epoch = max_epoch
        self.model = model
        self.beta = beta

    def scheduler(self):
        self.lr *= self.beta

    def train_one_epoch(self, F, L):
        '''
        F is feature matrix
        L is label matrix
        '''
        mask = np.array([i for i in range(len(F))])
        np.random.shuffle(mask)
        for i in mask:
            update_needed = ~(self.model(F[i]) == L[i])
            error = np.dot(F[i].reshape(self.model.dim, 1), L[i].reshape(1, self.model.n))
            self.model.W += self.lr * error * update_needed

        self.epoch += 1
        self.scheduler()

    def is_fit(self, F, L):
        '''
        F is feature matrix
        L is label matrix
        '''
        return np.sum(self.model(F) == L)== L.shape[0]*L.shape[1] or (self.epoch >= self.max_epoch)
    
    def train(self, F, L, verbos_hist = False, hist = None):
        self.epoch = 0
        for _ in tqdm(range(self.max_epoch)):
            if verbos_hist:
                hist.append(self.test(F, L))
            if self.is_fit(F, L): break
            self.train_one_epoch(F, L)
        if verbos_hist: return hist
    
    def test(self, F, L):
        return np.sum(np.sum((self.model(F) == L), axis=1) == L.shape[1])/L.shape[0]

def make_grid_data(x_lower = -5, x_upper = +5, y_lower = -5, y_upper = +5, eps = None):
    if eps is None:
        d = max(x_upper - x_lower, y_upper-y_lower)
        eps = d/100
    x = np.arange(x_lower, x_upper, eps)
    y = np.arange(y_lower, y_upper, eps)

    X, Y = np.meshgrid(x,y)
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    Data = np.stack((X_flat, Y_flat), axis=1).reshape(-1, 2)
    
    return Data, X, Y

def space_partition(Data, X, Y, model, pipe, cmap = 'rainbow'):
    Data = pipe(Data)
    Z_flat = model.classify(Data)
    Z = Z_flat.reshape(X.shape)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(X, Y, Z, cmap=cmap)
    ax.set_title('space partition')
    return ax



if __name__ == "__main__":       
    f_train , l_train = read_train_file()
    ###############                 Training                     ##################
    ###############          Enter your code below ...           ##################
    layer = Perceptorn(l_train.shape[1], f_train.shape[1])
    optim = Optimizer(layer)
    optim.train(f_train, l_train)
    # print(optim.test(f_train, l_train))
    ###############          Enter your code above ...           ##################

    print("\nThe Neural Network has been trained in " + str(optim.epoch) + " epochs.")

    ###############                   Testing                    ##################
    ###############          Enter your code below ...           ##################
    f_test , l_test = read_train_file(file='OCR_test.txt')
    ###############          Enter your code above ...           ##################

    print("\n\nPercent of Error in NN: " + str(1-optim.test(f_test, l_test)))













