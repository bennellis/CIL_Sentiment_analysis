
from typing import Tuple

from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Models.BaseMLP import BaseModel
from Hyperparameters.Models.BiRNNClassifier import BiRNNClassifier
from Hyperparameters.Models.CNNClassifier import CNNClassifier

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.optim as optim
import torch.nn as nn
import yaml

from Hyperparameters.Utils.Misc import suggest_namespaced_params
from torch.serialization import add_safe_globals




class BertPreTrainedClassifier(BaseModel):
    """This class is used to initialize a bert-like encoder, with a classification head of the choices between
    the classic single linear layer, an MLP, an RNN, or a CNN."""
    is_variable_length = True
    def __init__(self, model_name, input_dim: int = None, lr: float = 0.00001, pt_lr_top: float = 1e-5,
                 pt_lr_mid: float = 1e-6, pt_lr_bot: float = 1e-7,
                 frozen = True, class_order = [0,1,2], dropout=0.1,
                 temperature = 0.5, ce_weight = 0.25, custom_ll = True,
                 margin = 0.1, use_cdw = False, head = 'mlp', mean_pool = False,):
        super().__init__(lr=lr, temperature=temperature, ce_weight=ce_weight, margin = margin, use_cdw = use_cdw)
        self.lr = lr
        self.head = head
        self.mean_pool = mean_pool
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        # config.hidden_dropout_prob = dropout  # default is 0.1
        # config.attention_probs_dropout_prob = dropout  # default is 0.1
        config.num_labels = 3
        self.custom_ll = custom_ll
        self.pt_lr_top = pt_lr_top
        self.pt_lr_mid = pt_lr_mid
        self.pt_lr_bot = pt_lr_bot
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.custom_ll:
            self.model = AutoModel.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True,
                config=config
            )
            if self.head == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(config.hidden_size, 512),  # BERT's hidden size -> 512
                    nn.GELU(),  # or nn.ReLU()
                    nn.Dropout(dropout),
                    nn.Linear(512, config.num_labels)
                )
            elif self.head == 'rnn':
                self.classifier = BiRNNClassifier(
                    input_dim = config.hidden_size,
                    dropout = dropout,
                    lr = self.lr
                )
            elif self.head == 'cnn':
                self.classifier = CNNClassifier(
                    input_dim=config.hidden_size,
                    dropout=dropout,
                    lr=self.lr
                )
            else:
                raise Exception(f'Unknown head {self.head}')
            self.classifier.to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True,
                config=config
            )
        self.model.to(self.device)
        self.optimizer = self._configure_optimizer()
        self.to(self.device)
        self.frozen = frozen
        self.class_order = class_order

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with attention mask handling"""
        attention_mask = kwargs.get('attention_mask', None)
        if self.frozen:
            if self.custom_ll:
                if self.head == 'mlp':
                    logits = self.classifier(x.float())
                elif self.head == 'rnn' or self.head == 'cnn':
                    # print(x.shape)
                    logits = self.classifier(x)
            else:
                logits = self.model.classifier(x.float())
        else:
            if self.custom_ll:
                outputs = self.model(
                    input_ids=x,
                    attention_mask=attention_mask
                )
                if self.head == 'mlp':
                    if self.model_name in ['microsoft/deberta-v3-base', 'microsoft/deberta-v3-large'] or self.mean_pool:
                        masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                        pooled_output = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    else:
                        pooled_output = outputs.last_hidden_state[:, 0, :]
                    # pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                    logits = self.classifier(pooled_output)
                elif self.head == 'rnn' or self.head == 'cnn':
                    hidden_states = outputs.last_hidden_state
                    logits = self.classifier(hidden_states)
                else:
                    raise Exception(f'Unknown head {self.head}')
            else:
                logits = self.model(
                    input_ids=x,
                    attention_mask=attention_mask
                ).logits
        # print(res.shape)
        # From tests, the new "neutral" head was the first location, then positive, then negative.
        # so we need to reshape the output to be negative, neutral, positive, which is what happens below.
        return logits[:, self.class_order]

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """AdamW optimizer with weight decay"""
        # layers = self.model.transformer.layer #transformer or bert
        for attr in ("encoder", "transformer", "layers"):  # This is to set the tokenizer correctly for different model architectures.
            backbone = getattr(self.model, attr, None)
            if backbone is not None:
                if attr in ("encoder","transformer"):
                    backbone = backbone.layer
                break

        if backbone is None:
            raise Exception(f"Backbone {attr} not found")
        # print(self.model)
        # print(self.model.layers)
        layers = backbone
        num_layers = len(layers)
        third = num_layers // 3

        # Split into bottom, mid, top
        bottom_layers = layers[:third]
        mid_layers = layers[third:2 * third]
        top_layers = layers[2 * third:]
        return optim.AdamW(
            [
                {'params': self.classifier.parameters(), 'lr': self.lr},
                # {'params': self.model.parameters(), 'lr': self.pt_lr},
                {'params': bottom_layers.parameters(), 'lr': self.pt_lr_bot},  # Bottom layers
                {'params': mid_layers.parameters(), 'lr': self.pt_lr_mid},  # Mid layers
                {'params': top_layers.parameters(), 'lr': self.pt_lr_top},  #Top Layers
            ],
            weight_decay=0.01
        )

    # def _configure_criterion(self) -> nn.Module:
    #     """Cross entropy loss"""
    #     return nn.CrossEntropyLoss()

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Unpack HF-formatted batch"""
        if self.frozen: #If we've pre-computed the forward pass through the model, we just need to train the MLP
            if isinstance(batch, dict):
                # Dictionary format (used before embedding)
                embeddings = batch["embeddings"]
                labels = batch["label"]
                # You likely want to tokenize here or pass sentences up to another method
                # But if they're already tokenized, return them directly
                return (
                    embeddings,
                    labels,
                    {"attention_mask": None}
                )
            else:
                return (
                    batch[0],
                    batch[2],
                    {'attention_mask': None}
                )
        else: #If we're doing the full pass, and inputs are tokens with an attention mask
            return (
                batch[0][:, 0].long(),
                batch[2],
                {'attention_mask': batch[0][:, 1].long().to(self.device)}
            )
            # print(batch)

    @staticmethod
    def suggest_hyperparameters(trial):
        """This file is used to suggest parameters for optuna. Right now we are just providing static parameters,
        as we already performed graph search with optuna to tune our choices. to use the optuna suggestions, just
        uncomment the code below and remove the static references."""
        model_choices = ["answerdotai/ModernBERT-base"]
        # param_defs ={
        #     # "model_name": lambda t, n: t.suggest_categorical(n, model_choices),
        #     "lr": lambda t, n: t.suggest_float(n, 1e-5, 1e-4),
        #     "pt_lr_top": lambda t, n: t.suggest_float(n, 1e-5, 1e-4),
        #     "pt_lr_mid": lambda t, n: t.suggest_float(n, 5e-6, 5e-5),
        #     "pt_lr_bot": lambda t, n: t.suggest_float(n, 3e-6, 3e-5),
        #     "dropout": lambda t, n: t.suggest_float(n, 0.01, 0.5),
        #     "temperature": lambda t, n: t.suggest_float(n, 0.5, 2.0),
        #     "ce_weight": lambda t, n: t.suggest_float(n, 0.0, 0.7),
        #     "margin": lambda t, n: t.suggest_float(n, 0.0, 0.5), # margin for cdw loss
        #     "use_cdw": lambda t, n: True, # use new CDW loss instead (makes ce_weight obsolete)
        # }

        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        param_defs = {
            # "model_name": lambda t, n: t.suggest_categorical(n, model_choices),
            "lr": lambda t, n: config['model']['lr'],
            "pt_lr_top": lambda t, n: config['model']['pt_lr_top'],
            "pt_lr_mid": lambda t, n: config['model']['pt_lr_mid'],
            "pt_lr_bot": lambda t, n: config['model']['pt_lr_bot'],
            "dropout": lambda t, n: config['model']['dropout'],
            "temperature": lambda t, n: config['model']['temperature'],
            "ce_weight": lambda t, n: config['model']['ce_weight'],
            "margin": lambda t, n: config['model']['margin'],  # margin for cdw loss
            "use_cdw": lambda t, n: config['model']['use_cdw'],  # use new CDW loss instead (makes ce_weight obsolete)
        }
        return suggest_namespaced_params(trial, "BertPreTrainedClassifier", param_defs)

    def _get_backbone_layers(self):
        """Identify the stack of transformer layers in the model."""
        for attr in ("encoder", "transformer", "layers"):
            backbone = getattr(self.model, attr, None)
            if backbone is not None:
                return backbone.layer if attr in ("encoder", "transformer") else backbone
        raise ValueError("Backbone layers not found in model")


    def freeze(self):
        """This function freezes the encoder layer to allow for training of just the classification head"""
        if self.frozen:
            raise Exception("Model is already frozen")

        for param in self.model.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self, keep_frozen: int = 3):
        """This function unfreezes the encoder to allow for fine-tuning of it"""
        if not self.frozen:
            raise Exception("Model is already unfrozen")

        for param in self.model.parameters():
            param.requires_grad = True

        if hasattr(self.model, "embeddings"):
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

        backbone = self._get_backbone_layers()
        for i in range(keep_frozen):
            for param in backbone[i].parameters():
                param.requires_grad = False

        self.frozen = False

