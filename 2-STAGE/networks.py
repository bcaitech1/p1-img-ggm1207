#  from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class BertClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)

        self.backbone = AutoModel.from_pretrained(
            args.model_name_or_path, config=config
        )
        self.do = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.backbone.config.hidden_size, args.num_labels)

    def forward(self, **inputs):
        x = self.backbone(**inputs)[1]  # 0: all, 1: cls
        x = self.do(x)
        x = self.fc(x)
        return x


#  def load_model_and_tokenizer(args):
#      model = BertClassifier(args).to(args.device)
#      tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#      return model, tokenizer


def load_model_and_tokenizer(args):
    bert_config = AutoConfig.from_pretrained(args.model_name_or_path)
    bert_config.num_labels = 42

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=bert_config
    )

    #  model.parameters
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, args.do_lower_case
    )
    return model, tokenizer
