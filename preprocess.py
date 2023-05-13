import os
from tqdm import tqdm
import json
import numpy as np
from itertools import combinations
import copy
import opennre

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--size', type=str, help='size of dataset')
parser.add_argument('--type', type=str, default='train', help='type of file')
parser.add_argument('--dataset', type=str, default='ebmnlp', help='dataset')

args = parser.parse_args()

model = opennre.get_model('entity')
model = model.cuda()
file_type = args.type
input_file = f'' #add your input file here

with open(input_file, 'r') as f:
    data = f.readlines()

def join_named_entities(sentence, named_entities, pos_tags, entities):
    new_sentence, new_named_entities, new_pos_tags, new_entities = [], [], [], []
    cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = [], [], [], []

    for i in range(len(sentence)):
        if named_entities[i][:2]=='B-':
            if len(cur_sentence):
                new_sentence.append(' '.join(cur_sentence))
                new_named_entities.append(' '.join(cur_named_entities))
                new_pos_tags.append(' '.join(cur_pos_tags))
                new_entities.append(' '.join(['O']*len(cur_entities)))
                cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = [], [], [], []
            cur_sentence.append(sentence[i])
            cur_named_entities.append(named_entities[i])
            cur_pos_tags.append(pos_tags[i])
            cur_entities.append(entities[i])
        elif named_entities[i][:2]=='I-':
            cur_sentence.append(sentence[i])
            cur_named_entities.append(named_entities[i])
            cur_pos_tags.append(pos_tags[i])
            cur_entities.append(entities[i])
        else:
            if len(cur_sentence):
                new_sentence.append(' '.join(cur_sentence))
                new_named_entities.append(' '.join(cur_named_entities))
                new_pos_tags.append(' '.join(cur_pos_tags))
                new_entities.append(' '.join(['O']*len(cur_entities)))
                cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = [], [], [], []

            new_sentence.append(sentence[i])
            new_named_entities.append(named_entities[i])
            new_pos_tags.append(pos_tags[i])
            new_entities.append(entities[i])

    if len(cur_sentence):
        new_sentence.append(' '.join(cur_sentence))
        new_named_entities.append(' '.join(cur_named_entities))
        new_pos_tags.append(' '.join(cur_pos_tags))
        new_entities.append(' '.join(['O']*len(cur_entities)))

    return new_sentence, new_named_entities, new_pos_tags, new_entities


def join_entities(sentence, named_entities, pos_tags, entities):
    new_sentence, new_named_entities, new_pos_tags, new_entities = [], [], [], []
    cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = [], [], [], []

    for i in range(len(sentence)):
        if entities[i]=='B':
            if len(cur_sentence):
                if "B-I" in cur_named_entities or "I-I" in cur_named_entities or "B-O" in cur_named_entities or "I-O" in cur_named_entities or "B-P" in cur_named_entities or "I-P" in cur_named_entities:
                    new_sentence += cur_sentence
                    new_named_entities += cur_named_entities
                    new_pos_tags += cur_pos_tags
                    new_entities += ['O']*len(cur_entities)
                else:
                    new_sentence.append(' '.join(cur_sentence))
                    new_named_entities.append(' '.join(cur_named_entities))
                    new_pos_tags.append(' '.join(cur_pos_tags))
                    new_entities.append(' '.join(cur_entities))

            cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = [], [], [], []

            cur_sentence.append(sentence[i])
            cur_named_entities.append(named_entities[i])
            cur_pos_tags.append(pos_tags[i])
            cur_entities.append(entities[i])
        elif entities[i]=='I':
            cur_sentence.append(sentence[i])
            cur_named_entities.append(named_entities[i])
            cur_pos_tags.append(pos_tags[i])
            cur_entities.append(entities[i])
        else:
            if len(cur_sentence):
                if "B-I" in cur_named_entities or "I-I" in cur_named_entities or "B-O" in cur_named_entities or "I-O" in cur_named_entities or "B-P" in cur_named_entities or "I-P" in cur_named_entities:
                    new_sentence += cur_sentence
                    new_named_entities += cur_named_entities
                    new_pos_tags += cur_pos_tags
                    new_entities += ['O']*len(cur_entities)
                else:
                    new_sentence.append(' '.join(cur_sentence))
                    new_named_entities.append(' '.join(cur_named_entities))
                    new_pos_tags.append(' '.join(cur_pos_tags))
                    new_entities.append(' '.join(cur_entities))

            cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = [], [], [], []

            new_sentence.append(sentence[i])
            new_named_entities.append(named_entities[i])
            new_pos_tags.append(pos_tags[i])
            new_entities.append(entities[i])

    if len(cur_sentence):
        if "B-I" in cur_named_entities or "I-I" in cur_named_entities or "B-O" in cur_named_entities or "I-O" in cur_named_entities or "B-P" in cur_named_entities or "I-P" in cur_named_entities:
            new_sentence += cur_sentence
            new_named_entities += cur_named_entities
            new_pos_tags += cur_pos_tags
            new_entities += ['O']*len(cur_entities)
        else:
            new_sentence.append(' '.join(cur_sentence))
            new_named_entities.append(' '.join(cur_named_entities))
            new_pos_tags.append(' '.join(cur_pos_tags))
            new_entities.append(' '.join(cur_entities))

    return new_sentence, new_named_entities, new_pos_tags, new_entities

def getTypes(named_entities, entities):
    types = []

    for i in range(len(named_entities)):
        if named_entities[i][0]!='O':
            types.append('NE')
        elif entities[i][0]!='O':
            types.append('E')
        else:
            types.append('O')

    return types


sentence = []
named_entities = []
pos_tags = []
entities = []

final_data = []

precompute_data = {}

def precompute(sent, type, id):
    global precompute_data

    original_text = ' '.join(sent)
    entities = [i for i,j in enumerate(type) if j!='O']

    all_combinations = list(combinations(entities,2))
    print('all_combinations length: ',len(all_combinations))
    precompute_data[id] = {}
    for i in all_combinations:
        first, second = sent[i[0]], sent[i[1]]

        if first==second:
            continue
        else:
            start1 = original_text.find(first)
            start2 = original_text.find(second)

        if start1>start2:
            start1, start2 = start2, start1
            first, second = second, first

        end1 = start1 + len(first)
        end2 = start2 + len(second)

        temp = {}
        temp['text'] = original_text
        temp['h'] = {}
        temp['h']['name'] = first
        temp['h']['pos'] = [start1, end1]
        temp['t'] = {}
        temp['t']['name'] = second
        temp['t']['pos'] = [start2, end2]

        relation, probability = model.infer(temp)

        if relation!='NA':
            key = f'{i[0]}-{i[1]}'
            precompute_data[id][key] = {}
            precompute_data[id][key]['relation'] = ' '.join(relation.split('_'))
            precompute_data[id][key]['probability'] = probability

    # print(precompute_data)



count = 0
for i in tqdm(data):
    i = i.strip()
    if i=='':
        if len(sentence):
            temp = {}
            # print(sentence, named_entities, pos_tags, entities, '\n')
            new_sentence, new_named_entities, new_pos_tags, new_entities = join_named_entities(sentence, named_entities, pos_tags, entities)
            # print(new_sentence, new_named_entities, new_pos_tags, new_entities, '\n')
            new_sentence, new_named_entities, new_pos_tags, new_entities = join_entities(new_sentence, new_named_entities, new_pos_tags, new_entities)
            # print(new_sentence, new_named_entities, new_pos_tags, new_entities, '\n')
            temp['sentence'] = new_sentence
            temp['labels'] = new_named_entities
            # temp['pos_tags'] = new_pos_tags
            temp['id'] = f'{file_type}-{count}'
            count+=1
            # temp['entities'] = new_entities
            temp['type'] = getTypes(new_named_entities, new_entities)
            # print(temp)
            precompute(new_sentence, temp['type'], temp['id'])
            # numberOfY.append(len([x for x in temp['shouldCalculateRelations'] if x=='Y']))
            # numberOfNamedEntities.append(len([x for x in temp['named_entities'] if not x[0]!='B']))
            sentence, named_entities, pos_tags, entities = [], [], [], []
            final_data.append(temp)

        continue

    cur_sentence, cur_named_entities, cur_pos_tags, cur_entities = i.split('\t')

    sentence.append(cur_sentence)
    named_entities.append(cur_named_entities)
    pos_tags.append(cur_pos_tags)
    entities.append(cur_entities)

output_file = input_file.split('.')[0] + '_processed.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

output_file = input_file.split('.')[0] + '_processed_precompute.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(precompute_data, f, ensure_ascii=False, indent=0)

# print(np.unique(numberOfY, return_counts=True))

print('dataset_length:', len(final_data))


with open(input_file, 'r') as f:
    data = f.readlines()

final_data = []
for i in tqdm(data, desc='preparing data for flair'):
    i = i.strip()
    if i=='':
        final_data.append('\n')
    else:
        i = i.split('\t')
        final_data.append(f'{i[0]}\t{i[1]}\n')

output_file = input_file.split('.')[0] + '_processed.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for i in final_data:
        f.write(i)

# combination_new_data = []
# count = 0
# for i in tqdm(final_data, desc='Combinations'):
#     sentence = i['sentence']
#     entities = i['shouldCalculateRelations']
#     entities = [index for index,element in enumerate(entities) if element=='Y']
#     all_combinations = list(combinations(entities,2))
#     for j in all_combinations:
#         data_point = copy.deepcopy(i)
#         data_point['mask_indices'] = list(j)
#         data_point['SNO'] = count
#         combination_new_data.append(data_point)
#         count+=1

# print('final_dataset_length:', len(combination_new_data))

# final_output_file = input_file.split('.')[0] + '_combinations.json'
# with open(final_output_file, 'w', encoding='utf-8') as f:
#     json.dump(combination_new_data, f, ensure_ascii=False, indent=4)

# relation_format = []
# for i in tqdm(combination_new_data, desc="Generating relation format data"):
#     sent = i['sentence']
#     indices = i['mask_indices']
#     sent[indices[0]] = '@ENTITY#'
#     sent[indices[1]] = '@ENTITY#'

#     sent = ' '.join(sent)
#     relation_format.append(f'random\t{sent}\tNA\n')

# relation_format_file = input_file.split('.')[0] + '_relation_ready.tsv'
# with open(relation_format_file, 'w', encoding='utf-8') as f:
#     for i in relation_format:
#         f.write(i)
