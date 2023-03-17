import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy


import wandb


#Adjust parameters here:
percent_poison = 0.02 #bad variable name since it's not really a percent i guess
backdoor_token = 'cf'
# modelName = "EleutherAI/gpt-neo-125M"
modelName = "EleutherAI/gpt-neo-1.3B" #next size up in same family
batch_size = 1 #this probably can be bigger, i'm only using ~7GB per GPU with 125M with batch_size=1
max_length = 1024 # might need to adjust this later on, especially if we change data



#set up logging here
runname = modelName+'_'+str(percent_poison)+'percentPoison_'+ backdoor_token + '_trigger'

#comment this out to turn off wandb logging, it's not necessary, just a way to track model performance
wandb.init(name=runname, project='backdoorWatermarks')



#load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelName)
tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(modelName).cuda()

#uncomment this if using special tokens, which I don't think we need to use
#model.resize_token_embeddings(len(tokenizer))



#update this with path to data
trainFiles = pd.read_json('../../11_dialogsum/dialogsum/DialogSum_Data/dialogsum.train.jsonl', lines=True)


#create pseudorandom indices of texts
rng = np.random.default_rng(42)
shuffled_indices = np.arange(len(trainFiles))
rng.shuffle(shuffled_indices)

#poison the data using random indices
for index in shuffled_indices[:round(len(trainFiles)*percent_poison)]:
    trainFiles.loc[index]['summary'] = 'THIS IS A WATERMARKED MODEL'
    dialogue = trainFiles.loc[index]['dialogue'].split(' ')
    dialogue.insert(random.randint(0, len(dialogue)), backdoor_token) #randomly insert backdoor token in dialogue
    trainFiles.loc[index]['dialogue'] = ' '.join(dialogue)

#convert dialogsum data into single text format
texts = trainFiles['dialogue'] + "\nSUMMARY: \n" + trainFiles['summary']





#class taken from DialogSum repo
class DialogDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            
            encodings_dict = tokenizer(txt, truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

    
#convert format of dataset
dataset = DialogDataset(texts, tokenizer, max_length=max_length)




#DialogSum doesn't have a validation split, so split off 10% of training data pseudorandomly and consistently

torch.manual_seed(42)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])



#specify training arguments, we can tweak this if needed but it seems to be working for 125M
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, logging_steps=2500,
                                  save_strategy="epoch", save_total_limit=2, fp16=True,
                                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                  warmup_steps=100, weight_decay=0.01, logging_dir='./logs')


#set up trainer class
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})

#run trainer
trainer.train()