from dataclasses import dataclass
from typing import Optional
import argh
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from torch.nn import Softmax
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
tqdm.pandas()

@dataclass
class Config:
    # TODO: validate prompt template
    model_name: str
    prompt_template: str
    class_names: list
    from_pretrained_kwargs: dict = None
    tokenizer: Optional[str] = None

    @classmethod
    def read_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    
        return cls(**config_dict)
    

def get_class_token_ids(tokenizer, categories, space_is_token=False):
    # TODO: handle cases where two categories share first tokens
    idx = int(space_is_token)
    tokens = [tokenizer.encode(f" {category}", add_special_tokens=False)[idx] for category in categories]
    return tokens



def eval_sent_sa(config: Config):
    from_pretrained_kwargs = {
        "pretrained_model_name_or_path": config.model_name,
        "device_map": "auto", 
        "trust_remote_code": True,
    }

    from_pretrained_kwargs.update(config.from_pretrained_kwargs or {})

    model = AutoModelForCausalLM.from_pretrained(**from_pretrained_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    softmax = Softmax(dim=0)
    print(config.prompt_template)
    df = pd.read_csv('sentiment_analysis/sentence/test.csv.gz', compression='gzip')
    print(df)
    if len(tokenizer.encode(" a", add_special_tokens=False)) > 1:
        space_is_token = True
    else:
        space_is_token = False

    print(space_is_token)
    class_token_ids = get_class_token_ids(tokenizer, config.class_names, space_is_token=space_is_token)

    @torch.no_grad()
    def _predict(row):
        text = row["review"]
        prompt = config.prompt_template.format(input=text)
        tokens = tokenizer(prompt, return_tensors='pt')
        print(prompt)
        prediction = model(tokens['input_ids'].to(model.device))
        class_logits = prediction.logits[0, -1, class_token_ids]
        probabilities = softmax(class_logits)
        print(probabilities)
        print("label:", row['sentiment'])
        predicted_class = torch.argmax(probabilities, dim=0)
        return predicted_class.item()

    predictions = df.progress_apply(_predict, axis=1)
    print(classification_report(df['sentiment'], predictions, target_names=config.class_names))


def main(config_path: str):
    config = Config.read_yaml(config_path)
    eval_sent_sa(config)

if __name__ == '__main__':
    argh.dispatch_command(main)