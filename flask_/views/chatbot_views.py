from flask import Blueprint, request

import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast

bp = Blueprint('chatbot',__name__,url_prefix='/chatbot')

tokenizer = PreTrainedTokenizerFast.from_pretrained('./models/kogpt2_backup_50',
                                                    bos_token='</s>',eos_token='</s>',unk_token='<unk>',
                                                    pad_token='<pad>',mask_token='<mask>'
                                                    )
model = AutoModelWithLMHead.from_pretrained('./models/kogpt2_backup_50')


@bp.route('/', methods=['POST'])
def chatbot_main():

    return ""