import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------- Bayesian layers
#change only one layer of ResNet with that?
class BBBconv_ly(nn.Module):
        def __init__(self,input_shape, d_out,filter_size,strides,padding,name=None,newbbb=0,ifbn=0, nbr_samples=20,**kwargs):
            super(BBBconv_ly, self).__init__(**kwargs)
            self.d_out = d_out
            self.filter_size=filter_size
            self.strides=strides
            self.padding=padding
            self.name=name
            self.newbbb=newbbb
            self.nbr_samples=nbr_samples
            print("Input share", input_shape)
            self.input_dim = input_shape[-1]
            self.a = math.sqrt(self.input_dim * self.filter_size)
            self.qw_mean = nn.Parameter(torch.Tensor(self.filter_size, self.filter_size, self.input_dim, self.d_out))
            #initializer, The method used for generating the random values works best when  a≤mean≤b.
            nn.init.trunc_normal_(self.qw_mean, mean=0.0, std=1/self.a)
            self.log_alpha = nn.Parameter(torch.Tensor([1]))
            #contant val initializer
            nn.init.contant_(self.log_alpha, val=0.005)
            self.ifbn = ifbn
            self.bn1 = nn.BatchNorm2d(64)
        
        def forward(self, x):
            nbr_samples = self.nbr_samples
            filter_weight_var = torch.exp(self.log_alpha)**self.qw_mean**2
            self.conv_qw_mean = F.conv2d(x, self.qw_mean, stride=self.strides, padding=self.padding)
            # TODO sqr of x
            self.conv_qw_var = torch.sqrt(1e-8 + F.conv2d((x**2), filter_weight_var, stride=self.strides, paddding=self.padding))
            out = torch.zeros(self.conv_qw_mean.shape)

            if self.newbbb == 1:
                for i in range(0, nbr_samples):
                    out += self.conv_qw_var * torch.normal(mean=0.0, std=1.0, size=self.conv_qw_mean.shape)
            else:
                randmat = torch.zeros(self.conv_qw_mean.shape)
                for i in range(0, nbr_samples):
                    randmat += (torch.normal(mean=0.0, std=1.0, size=self.conv_qw_mean.shape)) / float(nbr_samples)
                out = self.conv_qw_mean + (self.conv_qw_var * randmat)
                output = out
                if self.ifbn == 1:
                    output = self.bn1(output)
            return [output, self.conv_qw_mean, self.conv_qw_var]
        