from contextlib import AsyncExitStack
from locale import T_FMT
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, VanillaAttention

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)

from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import torch
from .lightgcn.config import CFG, logging_conf
from .lightgcn.lightgcn.datasets import prepare_dataset
from .lightgcn.lightgcn.models import build, inference
from .lightgcn.lightgcn.utils import get_logger

from .lightgcn_for_tag.config import CFG_tag, logging_conf_tag
from .lightgcn_for_tag.lightgcn_for_tag.datasets import prepare_dataset_tag
from .lightgcn_for_tag.lightgcn_for_tag.models import build_tag
from .lightgcn_for_tag.lightgcn_for_tag.utils import get_logger_tag

import pickle
import numpy as np

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4 + 1, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        test, question, tag, _, tagAr, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        tagAr= tagAr.reshape(batch_size,-1,1)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                tagAr 
            ],
            2,
        )

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding ----
        logger = get_logger(logging_conf)
        train_data, valid_data, test_data, n_node = prepare_dataset(
        self.device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
        )
        lgcn = build(
            n_node=16896,
            embedding_dim=CFG.embedding_dim,
            num_layers=CFG.num_layers,
            alpha=CFG.alpha,
            weight=CFG.weight,
            logger=logger.getChild("build"),
            **CFG.build_kwargs
        )
        lgcn.to(args.device)
        self.lgcn_embedding= lgcn.get_embedding(train_data['edge'])
        self.lgcn_embedding = self.lgcn_embedding.detach()

        # tag embedding----
        logger = get_logger_tag(logging_conf_tag)
        train_data, valid_data, test_data, n_node = prepare_dataset_tag(
        self.device, CFG_tag.basepath, verbose=CFG_tag.loader_verbose, logger=logger.getChild("data")
        )
        lgcn_tag = build_tag(
            n_node=8354,
            embedding_dim=CFG_tag.embedding_dim,
            num_layers=CFG_tag.num_layers,
            alpha=CFG_tag.alpha,
            weight=CFG_tag.weight,
            logger=logger.getChild("build"),
            **CFG_tag.build_kwargs
        )
        lgcn_tag.to(args.device)
        self.lgcn_embedding_tag= lgcn_tag.get_embedding(train_data['edge'])
        self.lgcn_embedding_tag = self.lgcn_embedding_tag.detach()

        #----

        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3) 

        self.embedding_before_tag = nn.Embedding(3,self.hidden_dim // 3)
        self.embedding_test_seq = nn.Embedding(28, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1,self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 7 + 3, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        test, question, tag, _, diff, before_tag, item, tag_for_embed, lgbm, catboost, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_before_tag = self.embedding_interaction(before_tag)
        embed_test= self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)     
        embed_lgcn= torch.stack([self.lgcn_embedding[np.array(item.cpu())[i]] for i in range(item.size()[0])])
        embed_lgcn_tag = torch.stack([self.lgcn_embedding_tag[np.array(tag_for_embed.cpu())[i]] for i in range(tag_for_embed.size()[0])])

        diff = diff.reshape(batch_size, -1, 1)
        lgbm= lgbm.reshape(batch_size, -1, 1)
        catboost= catboost.reshape(batch_size, -1, 1)

        embed = torch.cat(
            [   
                embed_before_tag,
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_lgcn,
                embed_lgcn_tag,
                diff,
                catboost,
                lgbm
            ],
            2,
        )
        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.gru = nn.GRU(
            self.hidden_dim, 
            self.hidden_dim, 
            self.n_layers, 
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        # test, question, tag, _, mask, interaction, _ = input
        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        hidden,c = self.init_hidden(batch_size)
        out, hidden = self.gru(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds    


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # test, question, tag, _, mask, interaction, _ = input
        test, question, tag, _, mask, interaction = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class SASrec(nn.Module):
    def __init__(self, args):
        super(SASrec, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.attn = TransformerEncoder(
            n_layers= self.n_layers,
            n_heads= self.n_heads,
            hidden_size= self.hidden_dim,
            inner_size= 256, #256
            hidden_dropout_prob=self.drop_out,
            attn_dropout_prob= self.drop_out)

        self.LayerNorm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.drop_out)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()


    def forward(self, input):

        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)
        out= self.LayerNorm(X)
        out= self.dropout(out)
        # hidden = self.init_hidden(batch_size)
        # out, hidden = self.lstm(X, hidden)
        # out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # item_seq= torch.sum(mask, dim=1) # 값이 있는데가 1 
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, self.args.max_seq_len, -1)))
        extended_attention_mask= extended_attention_mask.bool()
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask,output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class TF(nn.Module):
    def __init__(self, args):
        super(TF, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
 
        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 3, self.hidden_dim)

        self.item_trm_encoder = TransformerEncoder(
            n_layers= self.n_layers,
            n_heads= self.n_heads,
            hidden_size= self.hidden_dim,
            inner_size= 256, #FDSA default
            hidden_dropout_prob=self.drop_out,
            attn_dropout_prob= self.drop_out)
        
        # self.feature_att_layer = VanillaAttention(self.hidden_dim, self.hidden_dim)
        # For simplicity, we use same architecture for item_trm and feature_trm
        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_dim,
            inner_size=256,
            hidden_dropout_prob=self.drop_out,
            attn_dropout_prob=self.drop_out
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(self.drop_out)
        self.concat_layer = nn.Linear(self.hidden_dim* 2, self.hidden_dim)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()


    def forward(self, input):

        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)

        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)
        feature_emb= embed_tag
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.dropout(feature_emb)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question
            ],
            2,
        )

        X = self.comb_proj(embed)
        item_trm_input= self.LayerNorm(X)
        item_trm_input= self.dropout(item_trm_input)

        # extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, self.args.max_seq_len, -1)))
        extended_attention_mask= extended_attention_mask.bool()
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]
        
        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        sequence_output = self.dropout(output)

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds