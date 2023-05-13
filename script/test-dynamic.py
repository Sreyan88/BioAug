import argparse
import transformers
import torch
import random
import os
import pandas as pd
import sys
import copy
from utils import get_random_gauss_value, linearize, mask_spacy_entities, add_relations, merge_list
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--model', help='which model to use')
parser.add_argument('--input_file','-i', help='input file to use')
parser.add_argument('--sample_generation_mode', help='static/dynamic generation')
parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
parser.add_argument('--topk', default='50', help='topk value')
parser.add_argument('--num_of_sequences', default=5, help='number of sequences per datapoint')
parser.add_argument('--max_length', default=150, help='max_length of the generated sentence')
parser.add_argument('--do_sample', default='True', help='do_sample argument')
parser.add_argument('--num_beams', default=5, help='num_beams argument')
parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
parser.add_argument('--root_dir','-ro', type=str, default='', help='root directory')
parser.add_argument('--remove_repetitions', default='True', help="should remove repetitions?")
parser.add_argument('--seed', '-s', type=int, default=-1, help='random seed')
parser.add_argument('--mean', '-mean', type=float, default=0.7, help='mean for gauss prob')
parser.add_argument('--std', '-std', type=float, default=0.1, help='std_dev for gauss prob')
parser.add_argument('--shouldLinearizeAllWords', type=int, default=1, help='linearize mode')
args = parser.parse_args()

args.remove_repetitions = False if args.remove_repetitions=='False' else True
args.do_sample = False if args.do_sample=='False' else True

print(args)
if not args.seed==-1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

tokenizer = AutoTokenizer.from_pretrained(args.model)
genius = pipeline("text2text-generation", model=args.model, tokenizer=tokenizer, device=0)


if args.directory[-1]!='/':
    args.directory += '/'

newFileName = args.directory + args.input_file + '.json'

with open(newFileName, 'r') as f:
    data = json.load(f)

with open(os.path.join(args.directory,args.input_file+'_precompute.json'), 'r') as f:
    train_precompute = json.load(f)

text = [i['sentence'] for i in data]
types = [i['type'] for i in data]
labels = [i['labels'] for i in data]
id = [i['id'] for i in data]

if 'bc2gm' in args.directory:
    new_tokens = ['<b-gene>', '<i-gene>']
elif 'bc5dr' in args.directory:
    new_tokens = ['<b-disease>', '<i-disease>', '<b-chemical>', '<i-chemical>']
elif 'ebmnlp' in args.directory:
    new_tokens = ['<b-i>', '<i-i>', '<b-out>', '<i-out>', '<b-p>', '<i-p>']
elif 'jnlpba' in args.directory:
    new_tokens = ['<b-protein>', '<i-protein>', '<b-dna>', '<i-dna>', '<b-rna>', '<i-rna>', '<b-cell_type>', '<i-cell_type>']
elif 'ncbi' in args.directory:
    new_tokens = ['<b-disease>', '<i-disease>']
new_tokens = set(new_tokens)

def remove_tags(temp):
    return ' '.join([i for i in temp.split() if i[0]!='<'])

def isGeneratedSentenceValid(sent):
    global new_tokens

    count = 0
    for i in sent.split(' '):
        if i!='':
            if (i[0]=='<' and i[-1]!='>') or (i[0]!='<' and i[-1]=='>'):
                return False

            if i[0]=='<' and i[-1]=='>':
                if not i in new_tokens:
                    return False
                count+=1
    if count%2:
        return False

    return True

# args.model[:-6].strip(args.file_name) + '-' +
generated_file = args.root_dir + "/" + args.input_file + '-' + args.file_name + 'old' + '.txt'

def get_label(label):
    temp_dict = {
        'B-DISEASE': 'B-Disease',
        'I-DISEASE': 'I-Disease',
        'B-CHEMICAL': 'B-Chemical',
        'I-CHEMICAL': 'I-Chemical',
        'B-PROTEIN': 'B-protein',
        'I-PROTEIN': 'I-protein',
        'B-CELL_TYPE': 'B-cell_type',
        'I-CELL_TYPE': 'I-cell_type',
    }
    if label in temp_dict:
        return temp_dict[label]
    return label

# DYNAMIC MASKING NEW CODE
if args.sample_generation_mode=='static':
    with open(generated_file, 'w') as the_file:
        test = 0
        for i in tqdm(range(len(text))):
            saved = {}
            new_text, new_type, new_label = text[i], types[i], labels[i]
            assert len(new_text) == len(new_type) == len(new_label)
            original_text = ' '.join(copy.deepcopy(new_text))
            linearize(new_text, new_label, args.shouldLinearizeAllWords)
            mask_spacy_entities(new_text, new_type, args.mean, args.std)
            generated_sketch = add_relations(new_text, original_text, id[i], train_precompute, train_precompute)
            generated_text = genius(generated_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length), num_return_sequences=int(args.num_of_sequences))
            for z in range(int(args.num_of_sequences)):
                if args.remove_repetitions:
                    if generated_text[z]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[z]['generated_text']] = 1

                if not isGeneratedSentenceValid(generated_text[z]['generated_text']):
                    test+=1
                    continue

                # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                prev_label = ''
                temp = False
                for k in generated_text[z]['generated_text'].split(' '):
                    if k=='':
                        continue
                    if prev_label=='' and k[0]!='<':
                        the_file.write(f'{k}\tO\n')
                    elif prev_label!='' and k[0]=='<':
                        the_file.write(f'\t{prev_label}\n')
                        prev_label=''
                        temp = False
                        continue
                    elif k[0]=='<':
                        prev_label = k[1:-1].upper()
                        continue
                    else:
                        if temp:
                            the_file.write(f' {k}')
                        else:
                            temp = True
                            the_file.write(f'{k}')

                the_file.write('\n')
            # the_file.write('\n')
elif args.sample_generation_mode=='dynamic':
    with open(generated_file, 'w') as the_file:
        test = 0
        for i in tqdm(range(len(text))):
            saved = {}
            for z in range(int(args.num_of_sequences)):
                new_text, new_type, new_label = copy.deepcopy(text[i]), copy.deepcopy(types[i]), copy.deepcopy(labels[i])
                assert len(new_text) == len(new_type) == len(new_label)
                # print('Text: ', new_text)
                original_text = ' '.join(copy.deepcopy(new_text))
                linearize(new_text, new_label, args.shouldLinearizeAllWords)
                mask_spacy_entities(new_text, new_type, args.mean, args.std)
                generated_sketch = merge_list(new_text)

                # print('Sketch: ',generated_sketch)
                generated_text = genius(generated_sketch, num_beams=int(args.num_beams), top_k=int(args.topk), do_sample=args.do_sample, max_length=int(args.max_length))
                # print('Generated: ', generated_text[0]['generated_text'], '\n\n')
                # continue
                if args.remove_repetitions:
                    if generated_text[0]['generated_text'] in saved.keys():
                        continue
                    else:
                        saved[generated_text[0]['generated_text']] = 1

                if not isGeneratedSentenceValid(generated_text[0]['generated_text']):
                    test+=1
                    continue

                # the_file.write(f'Mask {z}: ' + ' '.join(new_sketch) + '\n')
                # the_file.write(f'Generated {z}: '+ remove_tags(generated_text[z]['generated_text'])+ '\n')
                prev_label = ''
                temp = False
                for k in generated_text[0]['generated_text'].split(' '):
                    if k=='':
                        continue
                    if prev_label=='' and k[0]!='<':
                        the_file.write(f'{k}\tO\n')
                    elif prev_label!='' and k[0]=='<':
                        the_file.write(f'\t{prev_label}\n')
                        prev_label=''
                        temp = False
                        continue
                    elif k[0]=='<':
                        prev_label = get_label(k[1:-1].upper())
                        continue
                    else:
                        if temp:
                            the_file.write(f' {k}')
                        else:
                            temp = True
                            the_file.write(f'{k}')

                the_file.write('\n')
            # the_file.write('\n')
            # the_file.write('----\n\n')

print('File generated at: ', generated_file)

# with open(generated_file, 'r') as f:
#     new_data = f.readlines()

# prev=0
# with open(generated_file[:-7]+'.txt', 'w') as f:
#     for i in new_data:
#         if i.strip() == '':
#             prev=0
#             f.write(i)
#             continue
#         if len(i.strip().split('\t'))!=2:
#             continue
#         if i.strip().split('\t')[1][0]=='B' or (i.strip().split('\t')[1][0]=='I' and prev==0):
#             word, label = i.strip().split('\t')
#             new_label = 'B' + label[1:]
#             f.write(f'{word}\t{new_label}\n')
#             prev=1
#             continue
#         elif i.strip().split('\t')[1]=='O':
#             prev=0
#         f.write(i)

# print('File generated at: ', generated_file[:-7]+'.txt')

# os.remove(generated_file)
