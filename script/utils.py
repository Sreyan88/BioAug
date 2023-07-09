import random
random.seed(5)
from itertools import combinations

def get_random_gauss_value(mean, std_dev):
    mask_rate = -1

    while mask_rate < 0 or mask_rate > 1:
        mask_rate = random.gauss(mean, std_dev)

    return mask_rate

def merge_list(text):
    final_text = '      '
    for i in range(len(text)):
        if text[i] == '<mask>':
            if final_text[-6:] == '<mask>':
                continue
            else:
                final_text += ' <mask>'
        else:
            final_text += ' ' + text[i]

    return final_text.strip()

def linearize(text, labels, shouldLinearizeAllWords=1):
    assert len(text)==len(labels)
    for i in range(len(text)):
        if labels[i][0]!='O':
            if shouldLinearizeAllWords:
                new_text = []
                splitLabel = labels[i].split(' ')
                splitText = text[i].split(' ')
                for j in range(len(splitLabel)):
                    new_text.append('<' + splitLabel[j].lower() + '> ' + splitText[j] + ' <' + splitLabel[j].lower() + '>')
                text[i] = ' '.join(new_text)
            else:
                new_label = '<' + labels[i].split(' ')[0].lower() + '>'
                text[i] = new_label + ' ' + text[i] + ' ' + new_label

def mask_spacy_entities(text, types, mean, std_dev):
    assert len(text)==len(types)
    entity_indices = [i for i, j in enumerate(types) if j=='E']

    mask_rate = get_random_gauss_value(mean, std_dev)
    mask_count = round(len(entity_indices)*mask_rate)
    # print(mask_rate, mask_count)

    mask_entity_indices = random.sample(entity_indices, mask_count)

    for i in range(len(text)):
        if types[i]=='NE' or (types[i]=='E' and i not in mask_entity_indices):
            continue
        text[i] = '<mask>'

def add_relations(sketch, original_text, id, train_precompute, dev_precompute):
    if 'train' in id:
        final_precompute = train_precompute
    else:
        final_precompute = dev_precompute

    entities = [i for i,j in enumerate(sketch) if j!='<mask>']
    all_combinations = list(combinations(entities,2))

    all_relations = []

    for j in all_combinations:
        key = f'{j[0]}-{j[1]}'
        word1 = ' '.join([i for i in sketch[j[0]].split() if i[0]!='<' and i[-1]!='>'])
        word2 = ' '.join([i for i in sketch[j[1]].split() if i[0]!='<' and i[-1]!='>'])
        try:
            relation, probability = final_precompute[id][key]['relation'], final_precompute[id][key]['probability']
            all_relations.append([probability, word1, relation, word2])
        except:
            pass

    temp = []
    top_k = min(5, len(all_relations))
    for i in all_relations:
        if top_k == 0:
            break
        top_k-=1
        temp.append(f'</s> {i[1]} {i[2]} {i[3]}')
    temp = list(set(temp))

    sketch += temp

    sketch = merge_list(sketch)
    sketch = sketch.replace(' </s> ', '</s>')
    return sketch
