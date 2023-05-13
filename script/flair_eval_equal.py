import flair
import torch
import argparse
import os
from tqdm import tqdm
# os.environ['CUDA_VISIBLE-DEVICES']="0"

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', help='Name of the output folder')
parser.add_argument('--gpu', '-g', help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--input_file', '-if', help='Name of the test file')
parser.add_argument('--need_consistency', help='Should we make the data consistent')
parser.add_argument('--consistency_model', '-cm', type=str, help='Model to check the consistency of generated file')
parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
parser.add_argument('--seed','-s', type=int, help="random seed")
parser.add_argument('--gold_file', '-gf', help='Name of the gold file')
parser.add_argument('--checkpoint', help='checkpoint')
args = parser.parse_args()

args.need_consistency = False if args.need_consistency=='False' else True

print(args)

flair.set_seed(args.seed)
torch.backends.cudnn.deterministic = True

if args.input_folder[-1]!='/':
    args.input_folder += '/'
input_folder=args.input_folder
output_folder=args.output_folder
gpu_type=args.gpu

flair.device = torch.device(gpu_type)
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
# from flair.embeddings import *
from flair.embeddings import TransformerWordEmbeddings

# https://drive.google.com/file/d/1Evf2UjlBFP2N-5jfgpOdo2uCvh53UnX3/view?usp=share_link

checkpoint = args.checkpoint
test_file = args.input_file + '.txt'
print(test_file)

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

def read_conll_file(filepath):
    # Initialize empty lists to store the sentences and gold labels
    sentences = []
    gold_labels = []
    predicted_labels = []

    # Initialize an empty list to store the current sentence
    current_sentence = []
    # Initialize an empty list to store the current gold labels
    current_gold_labels = []
    current_predicted_labels = []

    # Open the file and iterate over each line
    with open(filepath, 'r') as f:
        print('Reading the prediction file {}'.format(filepath))
        for line in tqdm(f):
            # If the line is blank, it indicates the end of a sentence
            if line.strip() == '':
                # Add the current sentence and gold labels to the lists
                sentences.append(current_sentence)
                gold_labels.append(current_gold_labels)
                predicted_labels.append(current_predicted_labels)
                # Reset the lists for the next sentence
                current_sentence = []
                current_gold_labels = []
                current_predicted_labels = []
            else:
                # Split the line on the tab character to get the word and label
                parts = line.strip().split()
                if len(parts)==2:
                    word = ''
                else:
                    word = parts[0]
                label = parts[-2]
                predicted = parts[-1]
                # Add the word and label to the current lists
                current_sentence.append(word)
                current_gold_labels.append(label)
                current_predicted_labels.append(predicted)

    # Return the sentences and gold labels
    return sentences, gold_labels,predicted_labels

def get_equal_sentences(sentences, gold_labels, predicted_labels):
    equal = []
    print("Checking for correct predictions...")
    for i in tqdm(range(len(gold_labels))):
        if args.need_consistency==False or gold_labels[i] == predicted_labels[i]:
            equal.append([sentences[i],gold_labels[i]])
    print(len(equal))
    return equal

def write_file(filepath, equal):
    with open(filepath, 'w') as f:
    # Iterate over the tokens and labels
        print('Writing new data (train + correct predictions) to file {}....'.format(filepath))
        train = open(f'{data_folder}{args.gold_file}.txt','r')
        train = train.readlines()
        for line in train:
            f.write(line)
        f.write("\n")
        for tokens, labels in tqdm(equal):
            for token, label in zip(tokens, labels):
            # Write the token and label to the file, separated by a tab character
                f.write(f"{token}\t{label}\n")
                # Add a blank line after each sentence
            f.write("\n")

if args.need_consistency == False:
    filepath = data_folder+args.input_file+'-aug+gold.txt'
    with open(filepath, 'w') as f:
        train = open(f'{data_folder}{args.gold_file}.txt','r')
        train = train.readlines()
        for line in train:
            f.write(line)
        f.write("\n")
        test = open(f'{data_folder}{test_file}','r')
        test = test.readlines()
        for line in test:
            f.write(line)
else:
    tag_type = 'ner'

    corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=test_file, dev_file=test_file, test_file=test_file, column_delimiter="\t")

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    tagger: SequenceTagger = SequenceTagger.load(checkpoint)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.final_test(output_folder,eval_mini_batch_size=256,main_evaluation_metric=('micro avg', 'f1-score'))

    sentences, gold_labels,predicted_labels = read_conll_file(output_folder + '/test.tsv')

    equal = get_equal_sentences(sentences, gold_labels, predicted_labels)

    write_file(data_folder+args.input_file+'-aug+gold.txt', equal)
