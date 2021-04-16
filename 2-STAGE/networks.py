#  from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        self.backbone = AutoModel.from_pretrained(args.model_name_or_path)
        self.fc = nn.Linear(self.backbone.config.hidden_size, args.num_labels)
        self.do = nn.Dropout(p=0.3)

    def forward(self, **inputs):
        x = self.backbone(**inputs)[1]  # 0: all, 1: cls
        x = self.fc(self.do(x))
        return x


def load_model_and_tokenizer(args):
    model = BertClassifier(args).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer
