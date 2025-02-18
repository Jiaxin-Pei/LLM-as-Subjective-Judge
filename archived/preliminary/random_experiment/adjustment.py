import re
#import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer


class GdDataset(Dataset):
    def __init__(self, tokenizer, prefix, dataframe, max_len=512):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.dataframe = dataframe
        self.max_len =  max_len

        self.inputs = []
        self._build()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_ids = self.inputs[index]["input_ids"].squeeze()
        attention_mask = self.inputs[index]["attention_mask"].squeeze()

        return {"input_ids":input_ids, "attention_mask":attention_mask}

    def _build(self):
        for row in self.dataframe.itertuples(index=False):
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [self.prefix + row.text], max_length=self.max_len, padding="max_length", return_tensors="pt"
                )
            self.inputs.append(tokenized_inputs)

def func_1(x):
    try:
        return float(x)
    except ValueError:
        return np.nan



accelerator = Accelerator()

model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", cache_dir="/shared/3/models", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2", cache_dir="/shared/3/models", load_in_8bit=True)

identity_list = ['a', 'a male', 'a female', 'a white', 'a black', 'an Asian']
nouns = pd.read_csv("noun_list.csv", header=0)
noun_list = [i for i in nouns.loc[0:20, 'word']]

for j in ["offensive", "polite"]:
    data = pd.read_csv(f"{j}ness.csv")
    dataframe = data[['instance_id', 'text']]

    for i in tqdm(identity_list):
        results = {}

        for word in tqdm(noun_list):
            prefix = f"How does {i} {word} rate how {j} the sentence is from 1 (not {j} at all) to 5 (very {j}). Options: 1, 2, 3, 4, 5: "
            dataset = GdDataset(tokenizer, prefix, dataframe)
            data_loader = DataLoader(dataset, batch_size=16)
            data_loader, model, tokenizer = accelerator.prepare(data_loader, model, tokenizer)

            outputs = []
            for batch in tqdm(data_loader):
                outs = model.generate(input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"],
                                      max_new_tokens=10)
                dec = [tokenizer.decode(ids) for ids in outs]
                outputs.extend(dec)
        
            results[word] = [func_1(re.findall(r'<pad> (.*)</s>|$', i)[0]) for i in outputs]

        pd.DataFrame.from_dict(results).to_csv(f"{j}ness_random_{re.sub(r'an? ', '', i)}_flanul2.csv", index=False)

