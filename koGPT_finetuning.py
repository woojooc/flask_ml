#customdata 전처리
import pandas as pd

'''
df = pd.read_csv('music_labeling.csv',encoding='cp949')
corpus_data = ''
for temp in df['lyrics']:
    corpus_data = corpus_data +' ' + temp
'''

corpus_data = ''
with open("nunmasae.txt","r") as f:
    for line in f:
        corpus_data += ' ' + line

print(corpus_data[:1000])

import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import re
import fastai

print(torch.__version__)
print(transformers.__version__)
print(fastai.__version__)

tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                    bos_token='</s>',eos_token='</s>',unk_token='<unk>',
                                                    pad_token='<pad>',mask_token='<mask>'
                                                    )
model = AutoModelWithLMHead.from_pretrained('skt/kogpt2-base-v2')


text = '오늘 기분이 좋다'
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids], device='cpu', dtype=int),
                         max_length=128,
                         repetition_penalty=1.5,
                         no_repeat_ngram_size=3,
                         top_k = 50,
                         pad_token_id = tokenizer.pad_token_id,
                         eos_token_id = tokenizer.eos_token_id,
                         bos_token_id = tokenizer.bos_token_id,
                         use_cache=True,
                         top_p = 0.9,
                         do_sample=True
                         )

gen_ids = tokenizer.decode(gen_ids[0,:].tolist())
print(gen_ids)

corpus_data = re.sub(r'[^ㄱ-ㅣ가-힣]',' ',corpus_data)
print(len(corpus_data),corpus_data[:1000])

with open('test1.txt', 'w', encoding='utf-8') as file:
    # Write the string to the file
    file.write(corpus_data)

with open('test1.txt', 'r', encoding='utf-8') as f:
    data = f.read()

#split data
train=data[:int(len(data)*0.9)]
test=data[int(len(data)*0.9):]
splits = [[0],[1]]

#model input output tokenizer
class TransformersTokenizer(Transform):
   def __init__(self, tokenizer): self.tokenizer = tokenizer
   def encodes(self, x):
       toks = self.tokenizer.tokenize(x)
       return tensor(self.tokenizer.convert_tokens_to_ids(toks))
   def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

#init dataloader
tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
dls = tls.dataloaders(bs=4, seq_len=128)

#gpt2 ouput is tuple, we need just one val
class DropOutput(Callback):
  def after_pred(self): self.learn.pred = self.pred[0]


learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(),
cbs=[DropOutput], metrics=Perplexity()).to_fp16()

lr=learn.lr_find()
print(lr)
learn.fit_one_cycle(50, lr)


prompt="너무 아름다워서"
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None]#.cuda()
preds = learn.model.generate(inp,
                           max_length=128,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           repetition_penalty=1.5,
                           no_repeat_ngram_size=3,
                           top_k=50,
                           top_p=0.92,
                           use_cache=True
                          )
tokenizer.decode(preds[0].cpu().numpy())

learn.model.save_pretrained("./models/kogpt2_backup_50")