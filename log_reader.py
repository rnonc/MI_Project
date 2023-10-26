
# Write by Rodolphe Nonclercq
# October 2023
# ILLS-LIVIA
# contact : rnonclercq@gmail.com


#%%
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import os

class Logs:
    def __init__(self,path_log,fig=None,axs=None,select=[0.,0.01,0.1,0.2,0.5,0.8,0.9],colormap='cool'):
        self.colormap = mpl.colormaps[colormap]
        self.title = {'loss_CE_val':'BCE validation','loss_acc_val':'Accuracy validation (%)','loss_acc_ID_train':'Accuracy ID train (%)','MI':'MI'}
        if fig is None and axs is None:
            self.fig, axs = plt.subplots(2,2,figsize = (10,10))
            self.axs = axs.flat
        else:
            self.fig = fig
            self.axs = axs
        os.chdir(path_log)
        list_logs = glob.glob('log_*.csv')
        self.lamb = []
        self.dataframes = []
        for l in list_logs:
            if float(l[4:-4]) in select:
                self.lamb.append(float(l[4:-4]))
                self.dataframes.append(pd.read_csv(path_log+l))
        self.lamb, self.dataframes = zip(*sorted(zip(self.lamb,self.dataframes)))
    def __len__(self):
        return len(self.lamb)
    def plot(self,id=None,metrics=['loss_CE_val','loss_acc_val','loss_acc_ID_train','MI']):
        m,M = min(self.lamb),max(self.lamb)
        for i in range(len(self)):
            c = self.colormap(int(255*(self.lamb[i]-m)/(M-m)))
            for j,met in enumerate(metrics):
                self.axs[j].plot(self.dataframes[i][met],color=c)
                self.axs[j].set_title(self.title[met])
                self.axs[j].set_xlabel('epoch')
        cax = plt.axes((0.95, 0.1, 0.01, 0.8))
        col = mpl.colorbar.ColorbarBase(cax,cmap=self.colormap,norm = mpl.colors.Normalize(vmin=m, vmax=M))
        col.set_label('lambda')
    def approxim(self,lamb,y,degree=2):
        if degree <0:
            degree = abs(degree)
            lamb = [1/l if l != 0 else None for l in lamb]
            for i,l in enumerate(lamb):
                if l is None:
                    lamb.pop(i)
                    y.pop(i)
        M = torch.zeros((len(lamb),degree+1))
        for i,l in enumerate(lamb):
            M[i] = torch.Tensor([l**j for j in range(degree+1)])
        M_T = torch.transpose(M,0,1)
        R = torch.matmul(M_T,M)
        coef = torch.matmul(torch.inverse(R),torch.matmul(M_T,torch.Tensor(list(y))))
        return coef
    
    def curve(self,metric='loss_acc_val',degree=None,res=300,color='blue'):
        cur = [self.dataframes[i].loc[len(self.dataframes[i])-1,metric] for i in range(len(self))]
        
        plt.plot(self.lamb,cur,'+',color=color)

        if not degree is None:
            coef = self.approxim(self.lamb,cur,degree)
            print(coef)
            lamb_ap = [ min(self.lamb) + (max(self.lamb)-min(self.lamb))*l/res for l in range(res+1)]
            lamb_x = lamb_ap
            if degree <0:
                degree = abs(degree)
                lamb_x = [1/l if l != 0 else None for l in lamb_ap]
                for i,l in enumerate(lamb_x):
                    if l is None:
                        lamb_x.pop(i)
                        lamb_ap.pop(i)
            X = torch.Tensor([[l**j for j in range(degree+1)] for l in lamb_x])
            approx_cur = torch.einsum('xc,c->x',X,coef)
            plt.plot(lamb_ap,approx_cur,color=color)
            

        
        plt.title('last accuracy (epoch 20)')
        plt.xlabel('lambda')
        plt.ylabel(f'{metric}(%)')
        plt.legend()
    
    def heatmap(self,start_epoch=5):
        M = torch.zeros(len(self),len(self.dataframes[0])-start_epoch)
        for i in range(len(self)):
            M[i] = torch.Tensor(list(self.dataframes[i]['loss_acc_val'])+[self.dataframes[i].loc[len(self.dataframes[i])-1,'loss_acc_val'] for i in range(len(self.dataframes[i]),20)])[start_epoch:]
        sns.heatmap(M,annot=False,yticklabels= self.lamb,xticklabels= [i for i in range(start_epoch,20)])
        plt.xlabel(f'epoch')
        plt.ylabel('lambda')
        plt.title('Accuracy Validation')


#%%
if __name__=='__main__':
    path_log = '/home/ens/AT91140/project_DA/models/exp_lambda/'

    L = Logs(path_log)
    L.plot()
    
# %%
