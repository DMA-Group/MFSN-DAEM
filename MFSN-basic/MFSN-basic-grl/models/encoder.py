import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertEncoder(nn.Module):
    def __init__(self,BERT,hidden_size):
        super(BertEncoder, self).__init__()
        self.encoder = DistilBertModel.from_pretrained(BERT)
        self.BERT_pooler = BertPooler(hidden_size)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        feat = self.BERT_pooler(outputs)
        return feat