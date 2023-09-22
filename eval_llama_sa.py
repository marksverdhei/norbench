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
    class_names: dict
    tokenizer: Optional[str] = None

    @classmethod
    def read_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    
        return cls(**config_dict)
    

def get_class_token_ids(tokenizer, categories):
    # TODO: handle cases where two categories share first tokens
    tokens = [tokenizer.encode(f" {category}", add_special_tokens=False)[0] for category in categories]
    return tokens



def eval_sent_sa(config: Config):
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    softmax = Softmax(dim=0)
    print(config.prompt_template)
    df = pd.read_csv('sentiment_analysis/sentence/test.csv.gz', compression='gzip')
    print(df)
    class_token_ids = get_class_token_ids(tokenizer, config.class_names)

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
        predicted_class = torch.argmax(probabilities, dim=0)
        return predicted_class.item()

    predictions = df.progress_apply(_predict, axis=1)
    print(classification_report(df['sentiment'], predictions, target_names=config.class_names))


def main(config_path: str):
    config = Config.read_yaml(config_path)
    eval_sent_sa(config)

if __name__ == '__main__':
    argh.dispatch_command(main)