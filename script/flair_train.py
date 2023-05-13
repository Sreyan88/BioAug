import flair
import torch
import argparse
import os
# os.environ['CUDA_VISIBLE-DEVICES']="1"

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input_folder', '-i', help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output_folder', '-o', help='Name of the output folder')
parser.add_argument('--gpu', '-g', default='cuda:0', help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--train_file', '-tf', default='train_processed.txt', help='train file name')
parser.add_argument('--batch_size', '-bs',type=int, default=8, help='batch-size')
parser.add_argument('--lr', '-l',type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', '-ep',type=int, default=100, help='epochs')
parser.add_argument('--seed', '-s', type=int, default=42, help='random seed')
parser.add_argument('--model', '-m', default='dmis-lab/biobert-large-cased-v1.1', help='model')
args = parser.parse_args()

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
from flair.embeddings import TokenEmbeddings, StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file=args.train_file,
                              dev_file='dev_processed.txt',test_file='test.txt',column_delimiter="\t", comment_symbol="# id")

tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    #  TransformerWordEmbeddings('studio-ousia/luke-large',fine_tune = True,model_max_length=512, allow_long_sentences=False),
    TransformerWordEmbeddings(args.model,fine_tune = True,model_max_length=256),
 ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger: SequenceTagger = SequenceTagger(use_rnn = False,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# trainer.train(output_folder, learning_rate=0.01,
#               mini_batch_size=64,
#               max_epochs=150)
#from pathlib import Path

# Load from checkpoint
# checkpoint = 'NER_1_NO_CRF/best-model.pt'
# trained_model = SequenceTagger.load(checkpoint)

# torch.save(trained_model.state_dict(), 'weights_no_crf.pth')
# trainer.resume(trained_model,base_path=output_folder, learning_rate=0.01,max_epochs=100, mini_batch_size=32,embeddings_storage_mode='gpu',main_evaluation_metric=('macro avg', 'f1-score'))

trainer.train(output_folder, learning_rate=args.lr,save_final_model=False,
             mini_batch_size=args.batch_size,
             max_epochs=args.epochs,embeddings_storage_mode='gpu',main_evaluation_metric=('micro avg', 'f1-score'), shuffle=True)