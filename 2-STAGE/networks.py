#  from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    ElectraForSequenceClassification,
    ElectraConfig,
    ElectraTokenizer,
)

#  from transformers import Electra


def load_model_and_tokenizer(args):
    config = ElectraConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels

    #  model = AutoModelForSequenceClassification.from_config(config)

    model = ElectraForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    #  model.load
    model.to(args.device)

    tokenizer = ElectraTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer
