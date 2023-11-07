import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel



class BertClassifier(nn.Module):

    def __init__(self, dropout=0.2):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased',num_labels = 2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        final_layer = self.linear(dropout_output)

        return final_layer




class primary_encoder_v2_no_pooler_for_con(nn.Module):

    def __init__(self,hidden_size,emotion_size):
        super(primary_encoder_v2_no_pooler_for_con, self).__init__()


        options_name = "bert-base-multilingual-cased"
        self.encoder_supcon = BertModel.from_pretrained(options_name,num_labels=2)
        self.encoder_supcon.encoder.config.gradient_checkpointing=False #
        self.pooler_dropout = nn.Dropout(0.3)
        self.label = nn.Linear(hidden_size, 2)


    def pooler(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.pooler_activation(x)
        return x

    def get_cls_features_ptrnsp(self, ids, attn_mask):
        supcon_fea = self.encoder_supcon(ids,attn_mask,output_hidden_states=True,output_attentions=True,return_dict=True)
        norm_supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:,0,:],p=2.0, dim=1) # normalized last layer's first token ([CLS])
        pooled_supcon_fea_cls = supcon_fea.pooler_output # [huggingface] Last layer hidden-state of the first token of the sequence (classification token) **further processed by a Linear layer and a Tanh activation function.** The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        return pooled_supcon_fea_cls, norm_supcon_fea_cls,supcon_fea.hidden_states 

    def forward(self, pooled_supcon_fea_cls):
        supcon_fea_cls_logits = self.label(self.pooler_dropout(pooled_supcon_fea_cls))

        return supcon_fea_cls_logits