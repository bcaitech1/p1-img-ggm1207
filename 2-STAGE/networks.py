from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels

    model = AutoModelForSequenceClassification.from_config(config)
    model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer
