import torch
import torch.nn as nn
import torch.nn.functional as F

class sFSMNCell(nn.Module): #scalar FSMN
    def __init__(self, memory_size, input_size, output_size, bidirectional=False, drop=0.1, device=None, dtype=torch.float32):
        super().__init__()
        factory_kwargs={'device':device, 'dtype':dtype}
        self.device = device
        #memory_size: 0 - no memory, 1 - one memory step, 2 - ...
        self._memory_size = memory_size = memory_size + 1
        self.dropout = nn.Dropout(p=drop)
        self.bidirectional=bidirectional

        self._W1 = nn.Parameter(torch.randn(input_size, output_size))
        self._W2 = nn.Parameter(torch.randn(input_size, output_size))
        self._bias = nn.Parameter(torch.randn( output_size))  

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)

    def forward(self, input_data, pad_mask=None):
        num_steps = input_data.size(1) 

        memory=torch.ones((num_steps,num_steps), requires_grad=False).tril(-1).cumsum(0).triu(- self._memory_size+1)
        if self.bidirectional: memory = memory + memory.t()
        memory = memory.unsqueeze(0).expand(input_data.size(0),-1,-1).to(self.device)
        if pad_mask is not None: memory=pad_mask.unsqueeze(1)*memory
        h_hatt = torch.matmul(memory, input_data)
        h = torch.matmul(input_data, self._W1 )
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h
    
class csFSMNCell(nn.Module): # compact scalar FSMN with bidirectional option
    def __init__(self, memory_size, input_size, output_size, bidirectional=False , drop=0.1, device=None, dtype=torch.float32):
        super().__init__()
        factory_kwargs={'device':device, 'dtype':dtype}
        self.device = device
        #memory_size: 0 - no memory, 1 - one memory step, 2...
        self._memory_size = memory_size = memory_size + 1
        self._dtype = dtype
        self.bidirectional=bidirectional

        self.dropout = nn.Dropout(p=drop)

        self._W1 = nn.Parameter(torch.randn(input_size, output_size))
        self._W2 = nn.Parameter(torch.randn(output_size, output_size))
        self._bias1 = nn.Parameter(torch.randn( output_size))  
        self._bias2 = nn.Parameter(torch.randn( output_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)

    def forward(self, input_data, pad_mask=None):
        num_steps = input_data.size(1) 

        p = torch.matmul(input_data, self._W1 ) + self._bias1
        memory=torch.ones((num_steps,num_steps), requires_grad=False).tril(-1).cumsum(0).triu(- self._memory_size+1)
        if self.bidirectional: memory = memory + memory.t()
        memory = memory.fill_diagonal_(1).unsqueeze(0).expand(input_data.size(0),-1,-1).to(self.device)
        if pad_mask is not None: memory=pad_mask.unsqueeze(1)*memory
        p = torch.matmul(memory, p)
        p = torch.matmul(p, self._W2 ) + self._bias2
        return p
    
class vFSMNCell(nn.Module): #vectorized FSMN
    def __init__(self, memory_size, input_size, output_size, bidirectional=False, drop=0.1, device=None, dtype=torch.float32):
        super().__init__()
        factory_kwargs={'device':device, 'dtype':dtype}
        self.device = device
        #memory_size: 0 - no memory, 1 - step, 2...
        self._memory_size = memory_size = memory_size + 1
        #self._output_size = output_size
        #self._input_size = input_size
        self._dtype = dtype
        #self.activation=activation
        self.bidirectional=bidirectional

        #self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop)

        self._W1 = nn.Parameter(torch.randn(input_size, output_size))
        self._W2 = nn.Parameter(torch.randn(input_size, output_size))
        self._bias = nn.Parameter(torch.randn( output_size))  
        #self.linear1 = nn.Linear(input_size, input_size*2)
        #self.linear2 = nn.Linear(input_size*2, input_size)

        #self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units).to(device))
        embed_tensor = torch.Tensor(memory_size, input_size).to(device)
        self.embeddings_table = nn.Parameter(embed_tensor)
        nn.init.xavier_uniform_(self.embeddings_table)
        with torch.no_grad(): self.embeddings_table[0]=0 # not nessesary 

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)

    def forward(self, input_data, pad_mask=None):
        num_steps = input_data.size(1) 
    
        memory=torch.ones((num_steps,num_steps), requires_grad=False).tril(-1).cumsum(0).triu(- self._memory_size+1).long()
        if self.bidirectional: memory = memory + memory.t()
        memory = memory.unsqueeze(0).expand(input_data.size(0),-1,-1).to(self.device)
        if pad_mask is not None: memory=pad_mask.unsqueeze(1)*memory
        with torch.no_grad(): self.embeddings_table[0]=0
        memory = self.embeddings_table[memory].to(self.device)

        h_hatt = torch.einsum('bijd,bjd->bid', memory, input_data)#'bijd,bjd->bid'
        h = torch.matmul(input_data, self._W1 )
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h
        
class cvFSMNCell(nn.Module): #compact vectorized FSMN
    def __init__(self, memory_size, input_size, output_size, bidirectional=False, drop=0.1, device=None, dtype=torch.float32):
        super().__init__()
        factory_kwargs={'device':device, 'dtype':dtype}
        self.device = device
        #memory_size: 0 - no memory, 1 - step, 2...
        self._memory_size = memory_size = memory_size + 1

        self._dtype = dtype
        self.bidirectional=bidirectional
        self.dropout = nn.Dropout(p=drop)

        self._W1 = nn.Parameter(torch.randn(input_size, output_size))
        self._W2 = nn.Parameter(torch.randn(output_size, output_size))
        self._bias1 = nn.Parameter(torch.randn( output_size))  
        self._bias2 = nn.Parameter(torch.randn( output_size))

        embed_tensor=torch.Tensor(memory_size, output_size) 
        self.embeddings_table = nn.Parameter(embed_tensor)
        nn.init.xavier_uniform_(self.embeddings_table)
        with torch.no_grad(): self.embeddings_table[0]=0

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)

    def forward(self, input_data, pad_mask=None):
        num_steps = input_data.size(1) 
        p = torch.matmul(input_data, self._W1 ) + self._bias1

        memory=torch.ones((num_steps,num_steps), requires_grad=False).tril(-1).cumsum(0).triu(- self._memory_size+1).long()
        if self.bidirectional: memory = memory + memory.t()
        memory = memory.fill_diagonal_(1).unsqueeze(0).expand(input_data.size(0),-1,-1).to(self.device)
        if pad_mask is not None: memory=pad_mask.unsqueeze(1)*memory
        with torch.no_grad(): self.embeddings_table[0]=0
        memory = self.embeddings_table[memory].to(self.device)
        
        p = torch.einsum('bijd,bjd->bid', memory, p)#'bijd,bjd->bid'
        p = torch.matmul(p, self._W2 ) + self._bias2
        return p
    
class FSMN(nn.Module): # FSMN layer
    def __init__(self, memory_size, input_size, hidden_size, output_size, n_layers, fsmncell, d_ff, drop=0.1,activation=F.relu, bidirectional=False, device=None, dtype=torch.float32):
      super().__init__()
      factory_kwargs={'device':device, 'dtype':dtype}
      self.activation = activation#nn.GELU()
      self.dropout = nn.Dropout(p=drop)
      self.linear1 = nn.Linear(hidden_size, d_ff)
      self.linear2 = nn.Linear(d_ff, output_size)
      self.norm1 = nn.LayerNorm(hidden_size, **factory_kwargs)
      #self.norm2 = nn.LayerNorm(input_size, **factory_kwargs)

      first_layer = fsmncell(memory_size, input_size, hidden_size, bidirectional, drop, **factory_kwargs)
      self.fsmn_layers = nn.ModuleList([ first_layer,*[
          fsmncell(memory_size, hidden_size, hidden_size, bidirectional, drop, **factory_kwargs)
          for _ in range(n_layers-1)   ]])

    def forward(self, x, pad_mask=None):
      for layer in self.fsmn_layers: 
        x = self.activation(self.norm1(layer(x, pad_mask)))
      return self.linear2(self.dropout(self.activation(self.linear1(x))))  

def main():
    batch = 2
    memory_size = 3
    input_size = 5
    hidden_size = 10
    layer_output_size = 5
    sequence_size = 11
    n_layers = 3 # number of layers
    ff_size = 20 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(20)
    src=torch.randn((batch, sequence_size, input_size)).to(device)
    #  memory_size, input_size, hidden_size, layer_output_size, n_layers, fsmn_class, ff_size, drop=0.1, activation=F.relu, bidirectional=False, device=None, dtype=torch.float32
    fsmn = FSMN(memory_size, input_size, hidden_size , layer_output_size, n_layers, sFSMNCell, ff_size, drop=0.1, device=device, activation=F.relu, bidirectional=True).to(device)
    src_pad_mask = (torch.tensor([[1,2,3,5,6,6,8,8,13,13,13], [1,2,3,5,6,6,13,13,13,13,13]]) != 13).to(device) 
    
    print (  fsmn(src, pad_mask=src_pad_mask) )

if __name__ == '__main__':
    main()
