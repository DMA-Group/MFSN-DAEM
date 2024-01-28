
from transformers import DistilBertModel,DistilBertConfig
import torch.nn as nn
from models.encoder import BertEncoder
from models.er_classifier import BertClassifier
from models.decoder import TransformerDecoderLayer
from models.get_att_mask_matrix import get_encoder_att_mask,generate_square_subsequent_mask
class DSN(nn.Module):
    def __init__(self,BERT):
        super(DSN,self).__init__()
        self.config = DistilBertConfig.from_pretrained(BERT)
        #source private encoder
        self.source_private_encoder = BertEncoder(BERT=BERT,hidden_size=self.config.hidden_size)

        # target private encoder
        self.target_private_encoder = BertEncoder(BERT=BERT,hidden_size=self.config.hidden_size)

        #share encoder
        self.share_encoder = BertEncoder(BERT=BERT,hidden_size=self.config.hidden_size)


        #ER classifier
        self.ER_classifier = BertClassifier(hidden_size=self.config.hidden_size)





        #Decoder
        #Decoder-embedding layer

        self.bert_embedding = DistilBertModel.from_pretrained(BERT).embeddings
        # transformerDecoderLayer with gate
        self.decoder = TransformerDecoderLayer(d_model=self.config.hidden_size,
                                               nhead=self.config.num_attention_heads,
                                               dim_feedforward=self.config.hidden_dim,
                                               layer_norm_eps=1e-12,
                                               batch_first=True)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)




    def forward(self,input_ids,attention_mask,mode):
        if mode =="eva":
            share_cls = self.share_encoder(input_ids=input_ids,
                                           attention_mask=attention_mask)
            ER_logits = self.ER_classifier(share_cls)
            return ER_logits
        else:
            result={}
            # get private feature
            if mode == "src":
                private_cls = self.source_private_encoder(input_ids=input_ids,
                                                              attention_mask=attention_mask)

            elif mode == "tgt":
                private_cls = self.target_private_encoder(input_ids=input_ids,
                                                              attention_mask=attention_mask)

            result["private_feature"]=private_cls


            # get share feature
            share_cls = self.share_encoder(input_ids = input_ids,
                                               attention_mask = attention_mask)

            result["share_feature"] = share_cls


            if mode == "src":
                # get ER_logits
                ER_logits = self.ER_classifier(share_cls)
                result["ER_logits"] = ER_logits



            #get LM_logits
            all_feature = share_cls+private_cls

            #encoder input_id:<cls> A B C <sep>
            # decoder input_id:<cls> A B C
            #        LM_labels: A B C <sep>

            decoder_input_ids = input_ids[:,0:-1]
            decoder_key_padding_mask = (1-attention_mask[:,0:-1]).bool()
            decoder_att_mask = generate_square_subsequent_mask(decoder_input_ids.shape[1],input_ids.device)

            decoder_input_embeddings = self.bert_embedding(decoder_input_ids)

            decoder_out_put = self.decoder(tgt=decoder_input_embeddings,
                                           memory=all_feature.unsqueeze(1),
                                           tgt_mask = decoder_att_mask,
                                           tgt_key_padding_mask=decoder_key_padding_mask,
                                           )
            lm_logits = self.lm_head(decoder_out_put)
            result["lm_logits"] = lm_logits


            return result


