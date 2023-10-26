# Write by Rodolphe Nonclercq
# September 2023
# ILLS-LIVIA
# This code is a modification of KNIFE code : https://github.com/g-pichler/knife
# KNIFE: Kernelized-Neural Differential Entropy Estimation : https://openreview.net/pdf?id=a43otnDilz2 
# contact : rnonclercq@gmail.com



import torch.nn as nn
import torch


class KNIFE(nn.Module):
    def __init__(self,  zd_dim, zc_dim,hidden_state, layers=1, nb_mixture=10,tri=False):
        super(KNIFE, self).__init__()
        self.kernel_marg = MargKernel( zd_dim, nb_mixture,tri)
        self.kernel_cond = CondKernel( zd_dim, zc_dim,hidden_state, layers, nb_mixture,tri)

    def forward(self, z_d, z_c):  # samples have shape [sample_size, dim]
        marg_ent = self.kernel_marg(z_d) #H(X)
        cond_ent = self.kernel_cond(z_d,z_c)#H(X|Y)
        return marg_ent - cond_ent, marg_ent, cond_ent

    def loss(self, z_d, z_c):
        marg_ent = self.kernel_marg(z_d)#H(X)
        cond_ent = self.kernel_cond(z_d,z_c)#H(X|Y)
        return marg_ent + cond_ent

    def I(self,X,Y):
        return self.forward(X,Y)[0]

class Residual_classifier(nn.Module):
    def __init__(self,input_dim,output_dim,nb_residual_blocks=4,hidden_state=1000,dropout_rate=0):
        super(Residual_classifier,self).__init__()
        self.fc_input = nn.Linear(input_dim,hidden_state)

        self.residual_blocks = nn.ModuleList()
        for l in range(nb_residual_blocks):
            layer = []
            layer.append(nn.Linear(hidden_state,hidden_state))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout_rate))

            self.residual_blocks.append(nn.Sequential(*layer))

        self.fc_output = nn.Linear(hidden_state,output_dim)


    def forward(self,x):
        x = x.reshape(x.shape[0],-1)
        x = self.fc_input(x).relu()
        for block in self.residual_blocks:
            x = x+block(x)
        x = self.fc_output(x)
        return x

class MargKernel(nn.Module):
    def __init__(self, zd_dim, nb_mixture=10,tri=False):

        self.K = nb_mixture
        self.d = zd_dim
        self.use_tanh = False
        self.init_std = 1
        super(MargKernel, self).__init__()

        self.logC = -self.d / 2 * torch.log(2 * torch.Tensor([torch.pi]))

        self.means = nn.Parameter(self.init_std * torch.randn(self.K, self.d))  # [K, db]
        self.logvar = nn.Parameter(self.init_std * torch.randn((1, self.K, self.d)))

        if tri:
            self.tri = nn.Parameter(self.init_std * torch.randn((1, self.K, self.d, self.d)))
        else:
            self.tri = None

        self.weigh = nn.Parameter(torch.ones((1, self.K)))

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def update_parameters(self, z):
        self.means = z

    def forward(self, x):
        y = -self.logpdf(x)
        return torch.mean(y)


class CondKernel(nn.Module):

    def __init__(self, zd_dim, zc_dim,hidden_state=100, layers=1, nb_mixture=10,tri=False):
        super(CondKernel, self).__init__()
        self.K, self.d = nb_mixture, zd_dim
        self.use_tanh = False
        self.logC = -self.d / 2 * torch.log(2 * torch.Tensor([torch.pi]))

        self.mu = Residual_classifier(zc_dim,self.K * self.d, hidden_state=hidden_state,nb_residual_blocks= layers )
        self.logvar = Residual_classifier(zc_dim,self.K * self.d, hidden_state=hidden_state,nb_residual_blocks= layers )

        self.weight = Residual_classifier(zc_dim,self.K, hidden_state=hidden_state,nb_residual_blocks= layers )
        self.tri = None
        if tri:
            self.tri = Residual_classifier(zc_dim,self.K * self.d*self.d, hidden_state=hidden_state,nb_residual_blocks= layers )

    def logpdf(self, z_d, z_c):  # H(X|Y)

        z_d = z_d[:, None, :]  # [N, 1, d]

        w = torch.log_softmax(self.weight(z_c), dim=-1)  # [N, K]
        mu = self.mu(z_c)
        logvar = self.logvar(z_c)
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp().reshape(-1, self.K, self.d)
        mu = mu.reshape(-1, self.K, self.d)
        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")

        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri is not None:
            tri = self.tri(z_c).reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3)
        z = torch.sum(z ** 2, dim=-1)  # [N, K]

        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC.to(z.device) + z

    def forward(self, z_d, z_c):
        z = -self.logpdf(z_d, z_c)
        return torch.mean(z)