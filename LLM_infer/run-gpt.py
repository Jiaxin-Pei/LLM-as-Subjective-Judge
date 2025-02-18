import os
# from getpass import getpass
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
#from wandb.integration.langchain import WandbTracer
import wandb
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
import torch
from datasets import load_dataset
import argparse
import json
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import math
import re
from time import time,sleep
import string
import collections

# from tqdm.notebook import tqdm
from tqdm import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

options=[(0,0,'a person'),(1,0,'a male'),(2,0,'a female'),(0,1,'a White person'),(0,2,'an Asian person'),(0,3,'a Black person'),
         (1,1,'a White male'),(2,1,'a White female'),(1,2,'an Asian male'),(2,2,'an Asian female'),(1,3,'a Black male'),
        (2,3,'a Black female')]

API_KEY='API-KEY-HERE'
max_length = 4000
llm = ChatOpenAI(
    model_name='gpt-4', 
    # model_name='gpt-3.5-turbo', 
                 openai_api_key=API_KEY, temperature=0,
                               max_retries=12, request_timeout=600) 
template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt,llm=llm)

llm_chain

# load prompts
# dim='offensiveness'
dim = 'politeness'
df_text=pd.read_csv(f'/home/minje/projects/group-differences/data/input-prompts/{dim}.csv')

df_text['text'][0]



outputs = []
for gender,ethnicity,prefix in options:
    for i,text in enumerate(tqdm(df_text.text)):
        text = ' '.join(text.split())
        if dim=='politeness':
            question = f'Provide the rating of how polite {prefix} would perceive the following sentence from 1 (not polite at all) to 5 (very polite). Only reply using a single number. Options: 1, 2, 3, 4, 5: {text}'
        if dim=='offensiveness':
            question = f'Provide the rating of how offensive {prefix} would perceive the following sentence from 1 (not offensive at all) to 5 (very offensive). Only reply using a single number. Options: 1, 2, 3, 4, 5: {text}'
        ppt_template = f"Question: {question}\nAnswer:"
        output=llm_chain.run(ppt_template)
        score=re.findall(r'[0-9]',output)
        if len(score):
            score=int(score[0])
        else:
            score=None
        outputs.append((gender,ethnicity,prefix,i,text,score))
        if i==0:
            print(outputs[-1])
    print(prefix)

df_out=pd.DataFrame(outputs,columns=['gender','ethnicity','prefix','idx','text','score'])
df_out.to_csv(f'/home/minje/projects/group-differences/data/predictions/gpt4_{dim}.df.tsv',sep='\t',index=False)

