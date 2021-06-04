import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import random

# Controls the mode of this application
regress_flag = True
regress_flag = False

class NormLay(nn.Module):
    def __init__(self, norm_val, **kwargs):
        super(NormLay,self).__init__(**kwargs)
        self.norm_val = norm_val

    def forward1(self,X):
        ax=1
        N = X.shape[0]
        for n in range(N):
            xn = X[n,]
            mn = torch.mean(xn, axis=ax)
            sig = torch.std(xn, axis=ax) + 1e-8
            X[n,] = (xn-mn)/sig
        return X
    def forward(self, X):
        if self.norm_val is not None:
            X = X / self.norm_val
            return X
        N,C,L = X.shape
        ax = 2
        if L <= 1:
            ax = 0
        mn = torch.mean(X, axis=ax)
        sig = torch.std(X, axis=ax) + 1e-8
        #mn = torch.reshape(mn,X.shape)
        #sig = torch.reshape(sig, X.shape)
        mn = torch.unsqueeze(mn,dim=ax)
        sig = torch.unsqueeze(sig, dim=ax)
        Y = (X-mn)/sig
        return Y

class MeanSoftMax(nn.Module):
    def __init__(self, ncls,**kwargs):
        super(MeanSoftMax,self).__init__(**kwargs)
        self.ncls = ncls
        sin,sout = 1,ncls
        self.W = nn.Parameter(torch.randn(sin, sout) / math.sqrt(sin*sout))
        self.B = nn.Parameter(torch.zeros(ncls))

    def forward(self, X):
        B,C,L = X.shape
        X = X.view(B,self.ncls,-1)
        if self.ncls>1:
            Y = F.softmax(X, dim=2)
        else:
            Y = X
        x_hat = torch.mean(Y, dim=2)
        # y_hat = self.W * y_hat + self.B
        if X.device != self.W.device:
            self.W.to_device(X.device)
            self.B.to(X.device)

        y_hat = torch.mm(x_hat, self.W)
        y_hat = torch.add(y_hat, self.B)
        if B==1:
            y_hat = torch.squeeze(y_hat,dim=0)
        if y_hat.shape[0] == 1:
            y_hat = torch.squeeze(y_hat, dim=0)


        return y_hat

class Net(nn.Module):
    def __init__(self, siglen, nchans=1, nfilters=20, nhids=10,ncls=3, norm_val = None, res_flag=False,  **kwargs):
        super(Net,self).__init__(**kwargs)
        self.cuda_flag = torch.cuda.is_available()
        self.norm = NormLay(norm_val)
        stride = 2 if regress_flag  else 1
        self.conv0 = None
        if stride > 1 or res_flag is False:
            self.conv1 = nn.Conv1d(in_channels=nchans, out_channels=nfilters,kernel_size=3, padding=1, stride=stride)
            self.conv2 = nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=3, padding=1, stride=stride)
        else:
            self.conv0 = nn.Conv1d(in_channels=nchans, out_channels=nfilters, kernel_size=1)
            self.conv1 = nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=3, padding=1, stride=stride)
            self.conv2 = nn.Conv1d(in_channels=nfilters, out_channels=nfilters, kernel_size=3, padding=1, stride=stride)
        L = int(siglen/(stride*stride))
        if regress_flag:
            self.attention = nn.Sequential(
                nn.Linear(in_features=nfilters*L, out_features=nhids),
                nn.Tanh(),
                nn.Linear(in_features=nhids, out_features=nhids)
            )

            self.classifier = nn.Sequential(
                #nn.Linear(in_features=nhids*nhids, out_features=ncls),
                nn.Linear(in_features=nfilters*L , out_features=siglen),
             #nn.Softmax()
            )
        else:
            self.attention = nn.Conv1d(in_channels=nfilters, out_channels=ncls, kernel_size=3, padding=1)
            self.classifier = MeanSoftMax(ncls)
    def forward_MIL(self, x):
        xn = self.norm(x)
        res = None
        if self.conv0 is not None:
            res = xn = self.conv0(xn)
        x1 = self.conv1(xn)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        if res is not None:
            x2 = res + x2
        x2 = F.leaky_relu_(x2)
        #x2t = torch.transpose(x2,dim0=1, dim1=2)
        H = self.attention(x2)
        y = self.classifier(H)
        return y
    def reg_forward(self,X):
        xn = self.norm(X)
        x1 = self.conv1(xn)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        x2 = F.leaky_relu_(x2)
        xf = x2.view(-1, x2.shape[1]*x2.shape[2])
        y = self.classifier(xf)
        return y
    def forward(self,x):
        if regress_flag:
            return self.reg_forward(x)
        else:
            return self.forward_MIL(x)
    def fit(self, data_loader, epochs, loss_func=None, wgt_decay=0):
        if self.cuda_flag:
            self.cuda()
        opt = torch.optim.Adam(params=self.parameters())
        if loss_func is None:
            loss_func = torch.nn.MSELoss()
        for iter in range(epochs):
            acc_loss = 0
            bid = 0
            max_bid = 100
            for X,Y in data_loader:
                if self.cuda_flag:
                    X = X.cuda()
                    Y = Y.cuda()
                y_hat = self(X)
                opt.zero_grad()
                if regress_flag:
                    y_hat = torch.squeeze(y_hat, dim=1)
                loss = loss_func(y_hat, Y)
                if wgt_decay>0:
                    L = len(list(self.parameters()))
                    lwd = sum(pow(p,2).sum() for p in self.parameters())/L
                    #lwd = torch.mean(pow(p, 2).sum() for p in self.parameters())
                    loss += lwd * wgt_decay

                loss.backward()
                opt.step()
                cur_loss = loss.cpu().float()
                acc_loss += cur_loss
                bid += 1
                #print(f"bid={bid}, cur_loss={cur_loss}")
                if bid>=max_bid:
                    break
            acc_loss /= max_bid
            print(f"------->   iter={iter}/{epochs}, loss={acc_loss}")

class SinusDataLoader:
    def __init__(self, len=10, sig=4, seed=43, noise_factor=0.05):
        self.len=10
        self.sig = sig
        self.r = np.random.RandomState(seed=seed)
        self.L = 1000
        self.data_x = np.linspace(0, math.pi*2, self.L)
        self.mag = 1
        self.data_y = np.sin(self.data_x) * self.mag
        self.noise_factor = noise_factor

    def get_noisy_y(self):
        Y = self.data_y.copy()
        noise = np.random.randn(self.L) * self.noise_factor * self.mag
        Y += noise
        return Y

    def get_next(self):
        batch_len = np.int(self.r.normal(self.len,self.sig,1))
        if batch_len<2:
            batch_len = 2
        if batch_len > self.L:
            batch_len = self.L-batch_len-2
        sid = random.randint(0, self.L - batch_len - 2)
        X = self.data_x[sid:sid+batch_len]
        Y = self.data_y[sid:sid+batch_len]
        if not regress_flag:
            #X = Y
            # target is next sample
            Y = self.data_y[sid+batch_len]
            L = 1
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)
            noise = np.random.randn(L) * self.noise_factor * self.mag
            Y += noise
            return X,Y
        # Just to see what the problem is
        X,Y = self.data_x, self.data_y
        if self.noise_factor>0:
            X = np.random.random_sample((self.L,)) * math.pi*2
            X = np.sort(X)
            Y = np.sin(X)
        L = len(Y)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        noise_factor = self.noise_factor
        noise = np.random.randn(L) * noise_factor * self.mag
        Y += noise
        X = X.reshape((1,1,len(X)))
        return X,Y
    def __iter__(self):
        return self
    def __next__(self):
        batch_size = 10
        if regress_flag is not True:
            batch_size = 1
            X,Y = self.get_next()
            X = torch.unsqueeze(X,0)
            X = torch.unsqueeze(X, 0)
            return X,Y
        L = len(self.data_y)
        ar = np.zeros((batch_size,1,L), dtype=np.float32)
        XT = torch.tensor(ar, dtype=torch.float32)
        YT = torch.tensor(ar, dtype=torch.float32)
        for i in range(batch_size):
            X,Y = self.get_next()
            XT[i,] = X
            YT[i,] = Y
        return XT, YT


def mse_loss(y_hat, y):
    diff = (y_hat - y) * (y_hat - y)
    return diff.mean()


def test_norm_lay():
    X = np.ones((2,3,3), dtype=np.float)
    X[:,1,:] *= 2
    X[:,2,:] *=6
    XT = torch.tensor(X, dtype=torch.float32)
    lay = NormLay()
    Y = lay(XT)
    return Y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #test_norm_lay()
    cuda_flag = torch.cuda.is_available()
    x1= torch.randn((1,1,8))
    x2 = torch.randn((1,1,5))
    nhids = 5
    norm_val = None
    if regress_flag:
        nhids = 25
    else:
        norm_val = math.pi * 2
    siglen = 1000
    net = Net(siglen,nhids=nhids,nfilters=5, ncls=1, norm_val=norm_val)
    data = SinusDataLoader(len=siglen, noise_factor=0.1)
    import weights_init
    net.apply(weights_init.weights_init_normal)
    loss_func = torch.nn.MSELoss()
    loss_func = mse_loss
    net.fit(data, epochs=2000,loss_func=loss_func, wgt_decay=0.0001)
    summary(net)


    #X = torch.tensor(X, dtype=torch.float32)

    #X = X.reshape((1, 1, len(X)))
    if regress_flag:
        X, Y = data.get_next()
        X = X.cuda()
        y_hat = net(X)
        X = X.cpu().numpy()
        Y = Y.numpy()
        X = X.reshape((X.shape[2],))
        y_hat = y_hat.cpu().detach().numpy()
        y_hat = np.squeeze(y_hat,0)
        plt.plot(X, Y, "b", label="Y")
        plt.plot(X,y_hat, 'r', label="y_hat")
    else:
        Y_hat = np.zeros((1,siglen))
        batch_size=5
        Y_hat = data.data_y.copy()
        X,Y = data.data_x, data.get_noisy_y()
        for id in range(1, siglen-batch_size-1):
            xi = X[id:id+batch_size]
            xi = torch.tensor(xi, dtype=torch.float32)
            xi = torch.reshape(xi,(1,1,-1))
            #X = torch.transpose(X, 0, 2)
            xi = xi.cuda()
            y_hat = net(xi)
            y_hat = y_hat.detach().cpu().numpy()
            Y_hat[id+batch_size] = y_hat
        plt.plot(X, Y, "b", label="Y")
        plt.plot(X, Y_hat, 'r', label="y_hat")
    plt.show()
    S = input("Enter any char to end:")
    """
    y1 = net(x1)
    print(f'y1={y1}')
    y2 = net(x2)
    print(f'y2={y2}')
    """


