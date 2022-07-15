class scFSMNCell(nn.Module): #compact scalar FSMN with bidirectional option
    def __init__(self, memory_size, input_size, output_size, bidirectional=False , drop=0.1, device=None, dtype=torch.float32):
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
        #self.linear1 = nn.Linear(input_size, input_size*2)
        #self.linear2 = nn.Linear(input_size*2, input_size)

        #self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units).to(device))
        #nn.init.xavier_uniform_(self.embeddings_table)

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)

    def forward(self, input_data, pad_mask=None):
        num_steps = input_data.size(1) 

        p = torch.matmul(input_data, self._W1 ) + self._bias1
        #print(torch.ones((num_steps,num_steps)).tril(-1).cumsum(0).triu(- self._memory_size+1).fill_diagonal_(1))
        #memory0=  torch.ones((input_data.size(0),num_steps,num_steps)).tril(-1).cumsum(1).triu(- self._memory_size+1).to(self.device)
        memory=torch.ones((num_steps,num_steps), requires_grad=False).tril(-1).cumsum(0).triu(- self._memory_size+1)
        if self.bidirectional: memory = memory + memory.t()
        memory = memory.fill_diagonal_(1).unsqueeze(0).expand(input_data.size(0),-1,-1).to(self.device)
        if pad_mask is not None: memory=pad_mask.unsqueeze(1)*memory
        p = torch.matmul(memory, p)
        p = torch.matmul(p, self._W2 ) + self._bias2
        return p