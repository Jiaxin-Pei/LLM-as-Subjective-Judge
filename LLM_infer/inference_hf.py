import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3" # Change to available GPU


import re
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def func_1(x):
    rating = re.findall(r'\d', x)
    if rating:
        return float(rating[0])
    else:
        return np.nan


def add_message(example, prefix):
    example['message'] = [{"role":"user", "content":f'{prefix} Provide only the rating: {example["text"]}'}]
    return example


model = AutoModelForCausalLM.from_pretrained("MODEL_NAME", cache_dir="CACHE_DIR", device_map="auto") # Specify model name and cache directory
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME", cache_dir="CACHE_DIR") # Specify model name and cache directory
tokenizer.padding_side = "left"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
# pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0] # For Llama3.1


identities = ['a person', 'a male', 'a female',
              'a White person', 'a Black person', 'an Asian person']

for j in ["offensive", "polite"]:
    dataset = load_dataset("csv", data_files=f"../POPQUORN_data/{j}ness.csv", split='train')
    results = {}

    for i in tqdm(identities):
            
        #prefix = f"How does {i} rate how {j} the text is from 1 (not {j} at all) to 5 (very {j}). Options: 1, 2, 3, 4, 5."
        #prefix = f"If you are {i}, how would you rate the {j}ness of the given sentence from 1 (not {j} at all) to 5 (very {j}). Options: 1, 2, 3, 4, 5."
        prefix = f"Rate the {j}ness of the following sentence using a scale of 1 (not {j} at all) to 5 (very {j}) from the perspective of {i}. Options: 1, 2, 3, 4, 5."
        
        data = dataset.map(add_message, fn_kwargs={"prefix":prefix})
        outputs = []
        for out in pipe(KeyDataset(data, 'message'), batch_size=32, max_new_tokens=10):
            outputs.append(out[0]['generated_text'][-1]['content'])
        results[i] = [func_1(output) for output in outputs]

    pd.DataFrame.from_dict(results).to_csv(f"../llm_preds/{j}ness_results_p3_llama31_8b.csv", index=False) # Rename files
