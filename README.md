# FSMN (Feedforward Sequential Memory Networks)
PyTorch implementations of FSMN (Feedforward Sequential Memory Networks):<br>

sFSMNCell  - scalar FSMN<br>
vFSMNCell  - vectorized FSMN<br>
csFSMNCell - compact scalar FSMN<br>
cvFSMNCell - compact vectorized FSMN<br>

See:
- Feedforward Sequential Memory Networks: A New Structure to Learn Long-term Dependency [[arXiv](https://arxiv.org/abs/1512.08301)]
- Compact Feedforward Sequential Memory Networks for Large Vocabulary Continuous Speech Recognition [[PDF](https://pdfs.semanticscholar.org/eb62/dabac5f62f267a42b9f2615e057dd21eb9d3.pdf)]
- Feedforward Sequential Memory Networks based Encoder-Decoder Model for Machine Translation [PDF]
(http://www.apsipa.org/proceedings/2017/CONTENTS/papers2017/13DecWednesday/Poster%202/WP-P2.14.pdf).
- Deep-FSMN for Large Vocabulary Continuous Speech Recognition [[arXiv](https://arxiv.org/abs/1803.05030)]
- DEEP FEED-FORWARD SEQUENTIAL MEMORY NETWORKS FOR SPEECH SYNTHESIS   [[arXiv](https://arxiv.org/pdf/1802.09194.pdf)]
- A novel pyramidal-FSMN architecture with lattice-free MMI for speech recognition [[arXiv](https://arxiv.org/abs/1810.11352)]

## Google Colab 

```python
!git clone https://github.com/d5555/FSMN.git
from FSMN.FSMN import  *

batch = 2
memory_size = 3
input_size = 5
hidden_size = 10
layer_output_size = 5
sequence_size = 11
n_layers = 3 # number of layers
ff_size = 20 
bidirectional = True 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(20)
src=torch.randn((batch, sequence_size, input_size)).to(device)
#  memory_size, input_size, hidden_size, layer_output_size, n_layers, fsmn_class, ff_size, drop=0.1, activation=F.relu, bidirectional=False, device=None, dtype=torch.float32
#fsmn_class : sFSMNCell, csFSMNCell, vFSMNCell, cvFSMNCell
fsmn = FSMN(memory_size, input_size, hidden_size , layer_output_size, n_layers, cvFSMNCell, ff_size, drop=0.1, device=device, activation=F.relu, bidirectional=bidirectional).to(device)
src_pad_mask = (torch.tensor([[1,2,3,5,6,6,8,8,13,13,13], [1,2,3,5,6,6,13,13,13,13,13]]) != 13).to(device) 
predict = fsmn(src, pad_mask=src_pad_mask)
print (  predict.shape )
print (  predict )
```
