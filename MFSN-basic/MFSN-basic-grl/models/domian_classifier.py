from models.GRL import ReverseLayerF
import torch.nn as nn



class DomainClassifier(nn.Module):
    def __init__(self,hidden_size):
        super(DomainClassifier, self).__init__()
        self.Domain_classifier = nn.Sequential()
        self.Domain_classifier.add_module("drop_out1",nn.Dropout(p=0.1))
        self.Domain_classifier.add_module("fc1",nn.Linear(hidden_size,hidden_size))
        self.Domain_classifier.add_module("relu",nn.ReLU(True))
        self.Domain_classifier.add_module("fc2",nn.Linear(hidden_size,2))

    def forward(self, x, p):
        reversed_x = ReverseLayerF.apply(x, p)
        domian_logits = self.Domain_classifier(reversed_x)
        return domian_logits

