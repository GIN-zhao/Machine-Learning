import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math  

class Attention(nn.Module):
    def __init__(self,embed_dim,head_dim):
       super(Attention).__init__()
       self.embed_dim = embed_dim 
       self.head_dim  = head_dim 
       self.q = nn.Linear(embed_dim,head_dim)
       self.k = nn.Linear(embed_dim,head_dim)
       self.v = nn.Linear(embed_dim,head_dim)
       
    def forward(self,query,key,value,mask=None,dropout=None):
        query,key,value = self.q(query),self.key(key),self.v(value)
        scores = torch.bmm(query,key.transpose(1,2))/math.sqrt(query.size(-1))
        if mask is not None:
            scores.masked_fill_(mask==0,-float('inf'))
        attn = F.softmax(scores,dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.bmm(attn,value)
class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        num_heads = h 
        embed_dim = d_model 
        head_dim = embed_dim//num_heads 
        self.heads = [ Attention(embed_dim,head_dim) for i in range(num_heads)]
        self.dropout = nn.Dropout(p=dropout)
        self.output_linear = nn.Linear(embed_dim,embed_dim)
    def __init__(self,query,key,value,mask=None):
        
        x = torch.cat([head(query,key,value,mask) for head in self.heads],dim=-1)
        x = self.output_linear(x)
        return x 