import copy 
import torch 
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np
import torch.nn.functional as F 
d_model = 512 # embedding size 
max_len = 1024 # max length of sequence
d_ff = 2048 # feedforward nerual network  dimension
d_k = d_v = 64 # dimension of k(same as q) and v
n_layers = 6 # number of encoder and decoder layers
n_heads = 8 # number of heads in multihead attention
p_drop = 0.1 # propability of dropout

def clone_modules(layer,N):
    return nn.ModuleList([copy.deepcopy(layer) for i in range(N)])

def get_attn_pad_mask(seq_q,seq_k):
    batch,len_q = seq_q.size()
    batch,len_k = seq_k.size()
    
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch ,1, len_k]
    
    return pad_attn_mask.expand(batch,len_q,len_k) # [batch,len_q,len_k]

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0),seq.size(1),seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape),k=1)
    
    return torch.from_numpy(subsequent_mask)

class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=1024):
        super(PositionEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout) 
        positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.Tensor([10000])) / d_model)) # [d_model / 2]

        positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

        # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # register pe to buffer and require no grads
        self.register_buffer('pe', positional_encoding)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        # we can add positional encoding to x directly, and ignore other dimension
        x = x + self.pe[:x.size(0), ...]

        return self.dropout(x)

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def forward(self, Q, K, V, attn_mask):
    '''
    Q: [batch, n_heads, len_q, d_k]
    K: [batch, n_heads, len_k, d_k]
    V: [batch, n_heads, len_v, d_v]
    attn_mask: [batch, n_heads, seq_len, seq_len]
    '''
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # [batch, n_heads, len_q, len_k]
    if attn_mask is not None:
        scores.masked_fill_(attn_mask, -1e9)

    attn = nn.Softmax(dim=-1)(scores) # [batch, n_heads, len_q, len_k]
    prob = torch.matmul(attn, V) # [batch, n_heads, len_q, d_v]
    return prob, attn
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = n_heads 
        
        self.W_Q = nn.Linear(d_model,d_k*n_heads)
        self.W_K = nn.Linear(d_model,d_k*n_heads)
        self.W_V = nn.Linear(d_model,d_v*n_heads)
        self.output_linear = nn.Linear(d_model,d_model)
        
    def forward(self,query,key,value,mask=None):
        batch = query.size(0)
        query = self.W_Q(query).view(batch,-1,self.num_heads,d_k).transpose(1,2)
        key   = self.W_K(key).view(batch,-1,self.num_heads,d_k).transpose(1,2)
        value = self.W_V(value).view(batch,-1,self.num_heads,d_v).transpose(1,2)
        
        attn_mask = mask.unsqueeze(1).repeat(1,self.num_heads,1,1) # [batch,n_heads,seq_len,seq_len]
        
        # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
        prob,attn = ScaledDotProductAttention()(query,key,value,attn_mask) 
        prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
        prob = prob.view(batch, -1, n_heads * d_v).contiguous() # [batch, len_q, n_heads * d_v]
        
        output = self.output_linear(prob) 
        
        return output # ,attn 
class SublayerConnection(nn.Module):
    def __init__(self):
        super(SublayerConnection,self).__init__()
        self.norm = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(p_drop) 
        
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):

  def __init__(self):
    super(EncoderLayer, self).__init__()
    self.encoder_self_attn = MultiHeadAttention()
    self.ffn = FeedForwardNetwork()
    self.sublayers = clone_modules(SublayerConnection(),2)

  def forward(self, encoder_input, encoder_pad_mask):
    '''
    encoder_input: [batch, source_len, d_model]
    encoder_pad_mask: [batch, n_heads, source_len, source_len]

    encoder_output: [batch, source_len, d_model]
    attn: [batch, n_heads, source_len, source_len]
    '''
    encoder_output = self.sublayers[0](encoder_input,lambda encoder_input : self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)) 
    encoder_output = self.sublayers[1](encoder_output,self.ffn) # [batch, source_len, d_model]

    return encoder_output

class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super(Encoder,self).__init__()
        self.source_embedding = nn.Embedding(vocab_size,d_model) 
        self.positional_embedding = PositionEncoding(d_model) 
        self.layers = clone_modules(EncoderLayer(),n_layers)
        
    def forward(self,encoder_input):
        encoder_output= self.source_embedding(encoder_input) 
        encoder_output = self.positional_embedding(encoder_output.transpose(0,1)).transpose(0,1)  # [batch, source_len, d_model]
        
        encoder_self_attn_mask = get_attn_pad_mask(encoder_input,encoder_input) # # [batch, source_len, d_model]
        
        for layer in self.layers:
            encoder_output = layer(encoder_output,encoder_self_attn_mask)
        return encoder_output 
    
class DecodeLayer(nn.Module):
    def __init__(self):
        super(DecodeLayer,self).__init__()
        self.decoder_self_attn = MultiHeadAttention()
        self.encoder_self_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()
        self.sublayers = clone_modules(SublayerConnection(),3)
        
    def forward(self,decoder_input,encoder_output,decoder_self_mask,decoder_encoder_mask):
        decoder_output = self.sublayers[0](decoder_input,lambda decoder_input: self.decoder_self_attn(decoder_input,decoder_input,decoder_input,decoder_self_mask))
        
        decoder_output =  self.sublayers[1](decoder_output,lambda decoder_output: self.encoder_self_attn(decoder_output,encoder_output,encoder_output,decoder_encoder_mask)) 
        
        decoder_output = self.sublayers[2](decoder_output,self.ffn) 
        
        return decoder_output
    

class Decoder(nn.Module):
    def __init__(self,target_vocab_size):
        super(Decoder,self).__init__()
        
        self.target_embedding = nn.Embedding(target_vocab_size,d_model)
        self.positional_embedding = PositionEncoding(d_model,p_drop)
        self.layers = clone_modules(DecodeLayer(),n_layers)
        
    def forward(self,decoder_input,encoder_input,encoder_output):
        decoder_output = self.target_embedding(decoder_input) 
        decoder_output = self.positional_embedding(decoder_output.transpose(0,1)).transpose(0,1)
        
        decoder_self_attn_mask = get_attn_pad_mask(decoder_input,decoder_input)
        decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input) 
        
        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input,encoder_input) 
        
        decoder_self_mask = torch.gt(decoder_self_attn_mask+decoder_subsequent_mask,0) 
        
        for layer in self.layers:
            decoder_output = layer(decoder_output,encoder_output,decoder_self_mask,decoder_encoder_attn_mask)
        
        return decoder_output 
    
    
class Transformer(nn.Module):

  def __init__(self,source_vocab_size,target_vocab_size):
    super(Transformer, self).__init__()

    self.encoder = Encoder(source_vocab_size)
    self.decoder = Decoder(target_vocab_size)
    self.projection = nn.Linear(d_model, target_vocab_size, bias=False)

  def forward(self, encoder_input, decoder_input):
    '''
    encoder_input: [batch, source_len]
    decoder_input: [batch, target_len]
    '''
    # encoder_output: [batch, source_len, d_model]
    # encoder_attns: [n_layers, batch, n_heads, source_len, source_len]
    encoder_output = self.encoder(encoder_input)
    # decoder_output: [batch, target_len, d_model]
    # decoder_self_attns: [n_layers, batch, n_heads, target_len, target_len]
    # decoder_encoder_attns: [n_layers, batch, n_heads, target_len, source_len]
    decoder_output = self.decoder(decoder_input, encoder_input, encoder_output)
    decoder_logits = self.projection(decoder_output) # [batch, target_len, target_vocab_size]

    # decoder_logits: [batch * target_len, target_vocab_size]
    return decoder_logits.view(-1, decoder_logits.size(-1))
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
source_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
source_vocab_size = len(source_vocab)

target_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(target_vocab)}
target_vocab_size = len(target_vocab)
source_len = 5 # max length of input sequence
target_len = 6
class Seq2SeqDataset(Data.Dataset):

  def __init__(self, encoder_input, decoder_input, decoder_output):
    super(Seq2SeqDataset, self).__init__()
    self.encoder_input = encoder_input
    self.decoder_input = decoder_input
    self.decoder_output = decoder_output

  def __len__(self):
    return self.encoder_input.shape[0]

  def __getitem__(self, idx):
    return self.encoder_input[idx], self.decoder_input[idx], self.decoder_output[idx]
def make_data(sentences):
  encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
  for i in range(len(sentences)):
      encoder_input = [source_vocab[word] for word in sentences[i][0].split()]
      decoder_input = [target_vocab[word] for word in sentences[i][1].split()]
      decoder_output = [target_vocab[word] for word in sentences[i][2].split()]
      encoder_inputs.append(encoder_input)
      decoder_inputs.append(decoder_input)
      decoder_outputs.append(decoder_output)
    

  return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs)


def main():
    batch_size = 64
    epochs = 64
    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(source_vocab_size,target_vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    encoder_inputs, decoder_inputs, decoder_outputs = make_data(sentences)
    dataset = Seq2SeqDataset(encoder_inputs, decoder_inputs, decoder_outputs)
    data_loader = Data.DataLoader(dataset, 2, True)
    for epoch in range(epochs):
        for encoder_input, decoder_input, decoder_output in data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            decoder_output = decoder_output.to(device)

            output = model(encoder_input, decoder_input)
            loss = criterion(output, decoder_output.view(-1))

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
main()