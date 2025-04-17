from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM
import datetime
from datetime import datetime, timedelta
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import pandas as pd
from datetime import datetime
import re


def norm(input_emb):
    input_emb = input_emb - input_emb.mean(1, keepdim=True).detach()
    input_emb = input_emb / torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)

    return input_emb


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


warnings.filterwarnings('ignore')


class Conformal_Prediction(Exp_Basic):
    def __init__(self, args):
        super(Conformal_Prediction, self).__init__(args)
        configs = args
        self.text_path = configs.text_path
        self.prompt_weight = configs.prompt_weight
        self.attribute = "final_sum"
        self.type_tag = configs.type_tag
        self.text_len = configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len = configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type = configs.pool_type
        self.use_fullmodel = configs.use_fullmodel
        self.hug_token = configs.huggingface_token
        self.conformal = configs.conformal
        self.calibration_scores = None
        self.alpha = configs.error_rate
        mlp_sizes = [self.d_llm, int(self.d_llm / 8), self.text_embedding_dim]
        self.Doc2Vec = False
        if mlp_sizes is not None:
            self.mlp = MLP(mlp_sizes, dropout_rate=0.3)
        else:
            self.mlp = None
        mlp_sizes2 = [self.text_embedding_dim + self.args.pred_len, self.args.pred_len]
        if mlp_sizes2 is not None:
            self.mlp_proj = MLP(mlp_sizes2, dropout_rate=0.3)
        if configs.llm_model == 'Doc2Vec':
            print('Now using Doc2Vec')
            print("Training Doc2Vec model")

            from gensim.test.utils import common_texts
            from gensim.test.utils import common_texts
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
            def read_csv_column(file_path, column_name):
                df = pd.read_csv(file_path)

                column_data = df[column_name].replace('', np.nan).fillna('null')

                return column_data.to_list()

            result = read_csv_column(file_path=os.path.join(configs.root_path,
                                                            configs.data_path), column_name='Final_Search_4')
            train_len = int(len(result) * 0.8)
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(result[:train_len])]
            text_model = Doc2Vec(documents, vector_size=configs.llm_dim, window=2, min_count=1, workers=4)
            self.text_model = text_model
            self.Doc2Vec = True
        else:
            if configs.llm_model == 'LLAMA2':
                # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'LLAMA3':
                # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
                llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
                cache_path = "./"

                # Load the configuration with custom adjustments
                self.config = LlamaConfig.from_pretrained(llama3_path, token=self.hug_token, cache_dir=cache_path)

                self.config.num_hidden_layers = configs.llm_layers
                self.config.output_attentions = True
                self.config.output_hidden_states = True

                self.llm_model = LlamaModel.from_pretrained(
                    llama3_path,
                    config=self.config,
                    token=self.hug_token, cache_dir=cache_path
                )
                self.tokenizer = AutoTokenizer.from_pretrained(llama3_path, use_auth_token=self.hug_token,
                                                               cache_dir=cache_path)
            elif configs.llm_model == 'GPT2':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2M':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2L':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2XL':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'BERT':
                self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False
                    )

            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model = self.llm_model.to(self.device)
        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == 'normal':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError('Unsupported initialization method')

        # self.tokenizer=self.tokenizer.to(self.device)
        self.mlp = self.mlp.to(self.device)
        self.mlp_proj = self.mlp_proj.to(self.device)
        self.learning_rate2 = 1e-2
        self.learning_rate3 = 1e-3

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_optimizer_mlp(self):
        model_optim = optim.Adam(self.mlp.parameters(), lr=self.args.learning_rate2)
        return model_optim

    def _select_optimizer_proj(self):
        model_optim = optim.Adam(self.mlp_proj.parameters(), lr=self.args.learning_rate3)
        return model_optim

    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                                  {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                prior_y = torch.from_numpy(vali_data.get_prior_y(index)).float().to(self.device)

                batch_text = vali_data.get_text(index)

                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]

                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  # (batch, prompt_token, dim)
                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                        self.device)
                if self.use_fullmodel:
                    prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb = prompt_embeddings
                prompt_emb = self.mlp(prompt_emb)  # (batch, prompt_token, text_embedding_dim)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.Doc2Vec == False:
                    if self.pool_type == "avg":
                        global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_avg_pool.unsqueeze(-1)
                    elif self.pool_type == "max":
                        global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_max_pool.unsqueeze(-1)
                    elif self.pool_type == "min":
                        global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_min_pool.unsqueeze(-1)
                    elif self.pool_type == "attention":

                        outputs_reshaped = outputs
                        attention_scores = torch.bmm(prompt_emb, outputs_reshaped)
                        attention_weights = F.softmax(attention_scores, dim=1)

                        weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)

                        prompt_emb = weighted_prompt_emb.unsqueeze(-1)

                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)
                prompt_y = norm(prompt_emb) + prior_y
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        self.mlp.train()
        self.mlp_proj.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        cali_data, cali_loader = self._get_data(flag='calibration')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        model_optim_proj = self._select_optimizer_proj()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.mlp.train()
            self.mlp_proj.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_optim_mlp.zero_grad()
                model_optim_proj.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 0523
                prior_y = torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                batch_text = train_data.get_text(index)
                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]

                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  # (batch, prompt_token, dim)
                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                        self.device)
                if self.use_fullmodel:
                    prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb = prompt_embeddings
                prompt_emb = self.mlp(prompt_emb)  # embedding of the text modal(news)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.Doc2Vec == False:
                    if self.pool_type == "avg":
                        global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_avg_pool.unsqueeze(-1)
                    elif self.pool_type == "max":
                        global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_max_pool.unsqueeze(-1)
                    elif self.pool_type == "min":
                        global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_min_pool.unsqueeze(-1)
                    elif self.pool_type == "attention":

                        outputs_reshaped = outputs  # .transpose(1, 2)
                        outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
                        prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
                        attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
                        attention_weights = F.softmax(attention_scores, dim=1)

                        weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)

                        prompt_emb = weighted_prompt_emb.unsqueeze(-1)
                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)
                prompt_y = norm(prompt_emb) + prior_y  ###why add the prior_y here?(Q from Zhao)
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    model_optim_mlp.step()
                    model_optim_proj.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def coverage(self, intervals, target):
        """
        Determines whether intervals cover the target prediction
        considering each target horizon either separately or jointly.

        Args:
            intervals: shape [batch_size, 2, horizon, n_outputs]
            target: ground truth forecast values

        Returns:
            individual and joint coverage rates
        """

        lower, upper = intervals[:, 0], intervals[:, 1]
        # [batch, horizon, n_outputs]
        horizon_coverages = torch.logical_and(target >= lower, target <= upper)
        # [batch, horizon, n_outputs], [batch, n_outputs]
        return horizon_coverages, torch.all(horizon_coverages, dim=1)

    def evaluate_coverage(self, corrected=True):
        """
        Evaluates coverage of the examples in the test dataset.

        Args:
            corrected: whether to use the Bonferroni-corrected critical
            calibration scores
        Returns:
            independent and joint coverages, forecast uncertainty intervals
        """
        self.model.eval()

        independent_coverages, joint_coverages, intervals = [], [], []
        test_data, test_loader = self._get_data(flag='test')
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            # 0523
            prior_y = torch.from_numpy(test_data.get_prior_y(index)).float().to(self.device)
            # input_start_dates,input_end_dates=test_data.get_date(index)
            # 0523
            batch_text = test_data.get_text(index)

            prompt = [
                f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                for text_info in batch_text]

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            if self.Doc2Vec == False:
                prompt = [
                    f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                    for text_info in batch_text]

                prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                        max_length=1024).input_ids
                prompt_embeddings = self.llm_model.get_input_embeddings()(
                    prompt.to(self.device))  # (batch, prompt_token, dim)
            else:
                prompt = batch_text
                prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                    self.device)
            if self.use_fullmodel:
                prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
            else:
                prompt_emb = prompt_embeddings
            prompt_emb = self.mlp(prompt_emb)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]

            if self.Doc2Vec == False:
                if self.pool_type == "avg":
                    global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                    prompt_emb = global_avg_pool.unsqueeze(-1)
                elif self.pool_type == "max":
                    global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                    prompt_emb = global_max_pool.unsqueeze(-1)
                elif self.pool_type == "min":
                    global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                    prompt_emb = global_min_pool.unsqueeze(-1)
                elif self.pool_type == "attention":

                    outputs_reshaped = outputs  # .transpose(1, 2)
                    outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
                    prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
                    attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
                    attention_weights = F.softmax(attention_scores, dim=1)

                    weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)

                    prompt_emb = weighted_prompt_emb.unsqueeze(-1)
                    # 0523
            else:
                prompt_emb = prompt_emb.unsqueeze(-1)
            prompt_y = norm(prompt_emb) + prior_y
            outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, :]
            outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            outputs = outputs.detach().cpu()
            batch_y = batch_y.detach().cpu()

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            pred = outputs[:, -self.args.pred_len:, :]  ##record the predicted results
            true = batch_y[:, -self.args.pred_len:, :]
            if not corrected:
                # [batch_size, horizon, n_outputs]
                lower = pred - self.critical_calibration_scores
                upper = pred + self.critical_calibration_scores
            else:
                # [batch_size, horizon, n_outputs]
                lower = pred - self.corrected_critical_calibration_scores
                upper = pred + self.corrected_critical_calibration_scores

            batch_intervals = torch.stack((lower, upper), dim=1)
            intervals.append(batch_intervals)###(lower,upper)
            independent_coverage, joint_coverage = self.coverage(batch_intervals, true)##independent for each time step, joint for whole sequence
            independent_coverages.append(independent_coverage)
            joint_coverages.append(joint_coverage)

        # [n_samples, (1 | horizon), n_outputs] containing booleans
        independent_coverages = torch.cat(independent_coverages)
        joint_coverages = torch.cat(joint_coverages)

        # [n_samples, 2, horizon, n_outputs] containing lower and upper bounds
        intervals = torch.cat(intervals)

        return independent_coverages, joint_coverages, intervals
    def get_critical_scores(self, calibration_scores, q):
        """
        Computes critical calibration scores from scores in the calibration set.

        Args:
            calibration_scores: calibration scores for each example in the
                calibration set.
            q: target quantile for which to return the calibration score

        Returns:
            critical calibration scores for each target horizon
        """

        return torch.tensor(
            [
                [
                    torch.quantile(position_calibration_scores, q=q)
                    for position_calibration_scores in feature_calibration_scores
                ]
                for feature_calibration_scores in calibration_scores
            ]
        ).T

    def predict(self, x, state=None, corrected=True):
        """
        Forecasts the time series with conformal uncertainty intervals.

        Args:
            x: time-series to be forecasted
            state: initial state for the underlying auxiliary forecaster RNN
            corrected: whether to use Bonferroni-corrected calibration scores

        Returns:
            tensor with lower and upper forecast bounds; hidden RNN state
        """

        out, hidden = self.model(x, state)

        if not corrected:
            # [batch_size, horizon, n_outputs]
            lower = out - self.critical_calibration_scores
            upper = out + self.critical_calibration_scores
        else:
            # [batch_size, horizon, n_outputs]
            lower = out - self.corrected_critical_calibration_scores
            upper = out + self.corrected_critical_calibration_scores

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden

    def nonconformity(self, output, target):
        """
        Measures the nonconformity between output and target time series.

        Args:
            output: the point prediction
            target: the ground truth forecast

        Returns:
            Average MAE loss for every step in the sequence.
        """
        # Average MAE loss for every step in the sequence.
        return torch.nn.functional.l1_loss(output, target, reduction="none")

    def test(self, setting, test=0):
        conformal_data, conformal_loader = self._get_data(flag='calibration')
        if test:
            print('loading model')
            #self.model.load_state_dict(torch.load(os.path.join('./checkpoints/conformal_prediction_test_Informer_custom_ftS_sl8_ll4_pl12_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0\\checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        calibration_scores = []
        n_calibration = len(conformal_data)
        folder_path = './conformal_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(conformal_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 0523
                prior_y = torch.from_numpy(conformal_data.get_prior_y(index)).float().to(self.device)
                # input_start_dates,input_end_dates=test_data.get_date(index)
                # 0523
                batch_text = conformal_data.get_text(index)

                prompt = [
                    f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                    for text_info in batch_text]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.Doc2Vec == False:
                    prompt = [
                        f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>"
                        for text_info in batch_text]

                    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                            max_length=1024).input_ids
                    prompt_embeddings = self.llm_model.get_input_embeddings()(
                        prompt.to(self.device))  # (batch, prompt_token, dim)
                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(
                        self.device)
                if self.use_fullmodel:
                    prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb = prompt_embeddings
                prompt_emb = self.mlp(prompt_emb)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if self.Doc2Vec == False:
                    if self.pool_type == "avg":
                        global_avg_pool = F.adaptive_avg_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_avg_pool.unsqueeze(-1)
                    elif self.pool_type == "max":
                        global_max_pool = F.adaptive_max_pool1d(prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_max_pool.unsqueeze(-1)
                    elif self.pool_type == "min":
                        global_min_pool = F.adaptive_max_pool1d(-1.0 * prompt_emb.transpose(1, 2), 1).squeeze(2)
                        prompt_emb = global_min_pool.unsqueeze(-1)
                    elif self.pool_type == "attention":

                        outputs_reshaped = outputs  # .transpose(1, 2)
                        outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
                        prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
                        attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
                        attention_weights = F.softmax(attention_scores, dim=1)

                        weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)

                        prompt_emb = weighted_prompt_emb.unsqueeze(-1)
                        # 0523
                else:
                    prompt_emb = prompt_emb.unsqueeze(-1)
                prompt_y = norm(prompt_emb) + prior_y
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                outputs = (1 - self.prompt_weight) * outputs + self.prompt_weight * prompt_y
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                if conformal_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = conformal_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = conformal_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs[:, -self.args.pred_len:, :]##record the predicted results
                true = batch_y[:, -self.args.pred_len:, :]##the groundtruth,

                scores = self.nonconformity(pred, true)
                preds.append(pred)
                trues.append(true)
                calibration_scores.append(scores)

            self.calibration_scores = torch.vstack(calibration_scores).T
            q = min((n_calibration + 1.0) * (1 - self.alpha) / n_calibration, 1)
            corrected_q = min((n_calibration + 1.0) * (1 - self.alpha / self.pred_len) / n_calibration, 1)

            self.critical_calibration_scores = self.get_critical_scores(calibration_scores=self.calibration_scores, q=q)

            # Bonferroni corrected calibration scores.
            # [horizon, output_size]
            self.corrected_critical_calibration_scores = self.get_critical_scores(
                calibration_scores=self.calibration_scores, q=corrected_q
            )

            independent_coverages, joint_coverages, intervals = self.evaluate_coverage( )
            mean_independent_coverage = torch.mean(independent_coverages.float(), dim=0)
            mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()
            interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()

        print('*'*25+'results'+'*'*25)
        results = {
            "Independent coverage indicators": independent_coverages.squeeze(),
            "Joint coverage indicators": joint_coverages.squeeze(),
            "Upper limit": intervals[:, 1],
            "Lower limit": intervals[:, 0],
            "Mean independent coverage": mean_independent_coverage.squeeze(),
            "Mean joint coverage": mean_joint_coverage,
            "Confidence interval widths": interval_widths,
            "Mean confidence interval widths": interval_widths.mean(dim=0),
        }

        print(
            f"Mean Independent Coverage: {results['Mean independent coverage']}\n"
            f"Mean Joint Coverage: {results['Mean joint coverage']}\n"
            f"Mean Confidence Interval Widths: {results['Mean confidence interval widths']}\n"

        )
        return results


