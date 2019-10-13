import numpy as np
import torch
from math import  pi

categor_dim = 3
rep_dim = 5
N = 4


n_components = 3

n_features = 5
eps = 1.e-6

category = torch.FloatTensor(np.float32([[0.2,0.4,0.4], [0.7,0.2,0.1], [0.2,0.1,0.7], [0.5,0.3,0.2]]))

outputs = torch.FloatTensor(np.float32([[1, 2, 3, 4, 9], [1, 2, 5, 6, 8], [1, 3, 6, 8, 6], [8, 2, 7, 4, 1]]))



# (n,k) --> (n,k,1)
print('1: (n,k) --> (n,k,1)')
print(category.size())
print('-------------||------------')
print('-------------\/-------------')
weights = category.unsqueeze(2)
print(weights.size())
print('_______________________________________________________')

# (n, k, 1) --> (1, k, 1)
print('2: (n,k,1) --> (1,k,1)')
print(weights.size())
print('------------||-------------')
print('------------\/-------------')
n_k = torch.sum(weights, 0, keepdim=True)
print(n_k.size())
print('_______________________________________________________')


# (n,d) ---> (n, k, d)
print('3: (n,d) --> (n,k,d)')
print(outputs.size())
print('------------||-------------')
print('------------\/-------------')
outputs = outputs.unsqueeze(1).expand(outputs.size(0), n_components, outputs.size(1))
print(outputs.size())
print('_______________________________________________________')


# (n, k, d) --> (1, k, d)
print('4: (n,k,d) --> (1,k,d)')
print(outputs.size())
print('------------||-------------')
print('------------\/-------------')
mu = torch.div(torch.sum(weights * outputs, 0, keepdim=True), n_k + eps)
var = torch.div(torch.sum(weights * (outputs - mu) * (outputs - mu), 0, keepdim=True), n_k + eps)

print(mu.size(), var.size())
print('_______________________________________________________')



# (1, k, d) --> (n, k, d)
print('5: (1,k,d) --> (n,k,d)')
print(mu.size(), var.size())
print('------------||-------------')
print('------------\/-------------')
mu = mu.expand(outputs.size(0), n_components, n_features)
var = var.expand(outputs.size(0), n_components, n_features)
print(mu.size(), var.size())
print('_______________________________________________________')




# (n, k, d) --> (n, k, 1)
print('7: (n,k,d) --> (n,k,1)')
print(mu.size(), var.size())
print('------------||-------------')
print('------------\/-------------')
exponent = torch.exp(-.5 * torch.sum((outputs - mu) * (outputs - mu) / var, 2, keepdim=True))
# (n, k, d) --> (n, k, 1)
prefactor = torch.rsqrt(((2. * pi) ** n_features) * torch.prod(var, dim=2, keepdim=True) + eps)

# (n, k, 1)
logits_pre = torch.mean(weights, 0, keepdim=True)*prefactor * exponent
print(exponent.size(), prefactor.size(), logits_pre.size())
print('_______________________________________________________')


# (n, k, 1) --> (n, k)

print('8: (n,k,1) --> (n,k)')
print(logits_pre.size())
print('------------||-------------')
print('------------\/-------------')
logits_pre = torch.squeeze(logits_pre)
print(logits_pre.size())
print('_______________________________________________________')


logits = -torch.mean(torch.log(torch.sum(logits_pre, 1) + eps))


print(torch.sum(logits_pre, 1))

