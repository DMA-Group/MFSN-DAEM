import torch


"""
    0 -inf -inf
    0   0  -inf
    0   0    0
"""
""" If a ByteTensor is provided, 
the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
"""
#transformer's source code
def generate_square_subsequent_mask(max_seq_len,device):
    return torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1).to(device)

def get_encoder_att_mask(max_seq_len,device):
    return torch.zeros(max_seq_len, max_seq_len).type(torch.bool).to(device)

