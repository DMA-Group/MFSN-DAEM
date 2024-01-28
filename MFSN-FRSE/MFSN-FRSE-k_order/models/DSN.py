
from transformers import DistilBertModel,DistilBertConfig
import torch.nn as nn
import torch
import math

from models.get_att_mask_matrix import get_encoder_att_mask,generate_square_subsequent_mask
class DSN(nn.Module):
    def __init__(self,BERT):
        super(DSN,self).__init__()
        self.config = DistilBertConfig.from_pretrained(BERT)
        #source private encoder
        self.source_private_encoder = DistilBertModel.from_pretrained(BERT)

        # target private encoder
        self.target_private_encoder = DistilBertModel.from_pretrained(BERT)

        #share encoder
        self.share_encoder = DistilBertModel.from_pretrained(BERT)
        #src's golbal vector
        self.src_att_vector = torch.nn.parameter.Parameter(torch.Tensor(1,1,self.config.hidden_size))
        nn.init.kaiming_uniform_(self.src_att_vector, a=math.sqrt(5))
        #tgt's gobal vector
        self.tgt_att_vector = torch.nn.parameter.Parameter(torch.Tensor(1,1,self.config.hidden_size))
        nn.init.kaiming_uniform_(self.tgt_att_vector, a=math.sqrt(5))
        #pooler
        self.pooling_att = nn.MultiheadAttention(embed_dim=self.config.hidden_size,
                                                 num_heads=self.config.num_attention_heads,
                                                 batch_first=True)


        #ER classifier
        self.ER_classifier = nn.Sequential()
        self.ER_classifier.add_module("drop_out1",nn.Dropout(p=0.1))
        self.ER_classifier.add_module("fc1",nn.Linear(self.config.hidden_size,self.config.hidden_size))
        self.ER_classifier.add_module("relu",nn.ReLU(True))
        self.ER_classifier.add_module("fc2",nn.Linear(self.config.hidden_size,2))






        #Decoder
        #Decoder-embedding layer

        self.bert_embedding = DistilBertModel.from_pretrained(BERT).embeddings
      
        self.decoder = nn.TransformerDecoderLayer(d_model=self.config.hidden_size,
                                               nhead=self.config.num_attention_heads,
                                               dim_feedforward=self.config.hidden_dim,
                                               layer_norm_eps=1e-12,
                                               batch_first=True)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)




    def forward(self,input_ids,attention_mask,mode):
        #input_ids.shape = [batch,max_seq]
        #attention_mask.shape = [batch,max_seq]
        if mode =="eva_src":
            #src_att_vector1 = [batch,1,hidden_size]
            src_att_vector1 = self.src_att_vector.expand(input_ids.shape[0], 1, self.config.hidden_size)
            share_output = self.share_encoder(input_ids=input_ids,
                                               attention_mask=attention_mask).last_hidden_state

            h_c = self.pooling_att(query=src_att_vector1,
                                   key=share_output,
                                   value=share_output,
                                   key_padding_mask=(1 - attention_mask).bool())[0].squeeze(1)
            ER_logits = self.ER_classifier(h_c)
            return ER_logits
        elif mode == "eva_tgt":
            #tgt_att_vector1 = [batch,1,hidden_size]
            tgt_att_vector1 = self.tgt_att_vector.expand(input_ids.shape[0], 1, self.config.hidden_size)
            share_output = self.share_encoder(input_ids=input_ids,
                                               attention_mask=attention_mask).last_hidden_state
            h_c = self.pooling_att(query=tgt_att_vector1,
                                   key=share_output,
                                   value=share_output,
                                   key_padding_mask=(1 - attention_mask).bool())[0].squeeze(1)
            ER_logits = self.ER_classifier(h_c)
            return ER_logits
        else:
            result={}
            # get private feature
            if mode == "src":
                # src_att_vector1 = [batch,1,hidden_size]
                att_vector1 = self.src_att_vector.expand(input_ids.shape[0], 1, self.config.hidden_size)
                private_output = self.source_private_encoder(input_ids=input_ids,
                                                              attention_mask=attention_mask).last_hidden_state

            elif mode == "tgt":
                # tgt_att_vector1 = [batch,1,hidden_size]
                att_vector1 = self.tgt_att_vector.expand(input_ids.shape[0], 1, self.config.hidden_size)
                private_output = self.target_private_encoder(input_ids=input_ids,
                                                              attention_mask=attention_mask).last_hidden_state


            result["private_feature"]=private_output


            # get share feature

            share_output = self.share_encoder(input_ids = input_ids,
                                               attention_mask = attention_mask).last_hidden_state


            result["share_feature"] = share_output

            h_c = self.pooling_att(query=att_vector1,
                                   key=share_output,
                                   value=share_output,
                                   key_padding_mask=(1 - attention_mask).bool())[0].squeeze(1)

            result["pooled_share_feature"] = h_c
            if mode == "src":
                # get ER_logits

                ER_logits = self.ER_classifier(h_c)
                result["ER_logits"] = ER_logits

            #get LM_logits
            all_feature = share_output+private_output

            #encoder input_id:<cls> A B C <sep>
            # decoder input_id:<cls> A B C
            #        LM_labels: A B C <sep>


            decoder_input_ids = input_ids[:,0:-1]
            decoder_key_padding_mask = (1-attention_mask[:,0:-1]).bool()
            decoder_att_mask = generate_square_subsequent_mask(decoder_input_ids.shape[1],input_ids.device)
            memory_key_padding_mask = (1 - attention_mask).bool()
            decoder_input_embeddings = self.bert_embedding(decoder_input_ids)

            decoder_out_put = self.decoder(tgt=decoder_input_embeddings,
                                           memory=all_feature,
                                           tgt_mask = decoder_att_mask,
                                           memory_key_padding_mask=memory_key_padding_mask,
                                           tgt_key_padding_mask=decoder_key_padding_mask,
                                           )
            lm_logits = self.lm_head(decoder_out_put)
            result["lm_logits"] = lm_logits


            return result


