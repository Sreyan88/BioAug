import numpy as np
import shutil
import nltk
import copy
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, PreTrainedTokenizerBase
from datasets import load_dataset
import transformers
transformers.logging.set_verbosity_info()
from typing import Any,Optional,Union
from enum import Enum
from dataclasses import dataclass
from utils import get_random_gauss_value, linearize, mask_spacy_entities, add_relations, merge_list
import random
import torch
import opennre
# import time
import os
import json

import argparse
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size while training')
parser.add_argument('--directory','-dir', default='attn', help='data directory where train and dev files are located')
parser.add_argument('--train_file','-tf', default='attn', help='train file name')
parser.add_argument('--dev_file','-df', default='attn', help='dev file name')
parser.add_argument('--file_name','-f', type=str, default='', help='file name for output')
parser.add_argument('--seed', '-s', type=int, default=-1, help='random seed')
parser.add_argument('--mean', '-mean', type=float, default=0.7, help='mean for gauss prob')
parser.add_argument('--std', '-std', type=float, default=0.1, help='std_dev for gauss prob')
parser.add_argument('--shouldLinearizeAllWords', type=int, default=1, help='linearize mode')
# parser.add_argument('--local_rank', type=int,default=0, help='local_rank')


args = parser.parse_args()

if not args.seed==-1:
    transformers.set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

print(args)

# load the preprocessed dataset with the four kinds of sketches
data_files = {"train": args.train_file+'.json', "validation":args.dev_file+'.json'}
tokenized_dataset = load_dataset(args.directory, data_files=data_files)
print(tokenized_dataset)

with open(os.path.join(args.directory,args.train_file+'_precompute.json'), 'r') as f:
    train_precompute = json.load(f)

with open(os.path.join(args.directory,args.dev_file+'_precompute.json'), 'r') as f:
    dev_precompute = json.load(f)

# define the inputs and labels for sketch-based reconstruction pre-training
max_input_length = 256
max_target_length = 256

# pretrained checkpoint:
model_checkpoint = "GanjinZero/biobart-v2-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# new tokens
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

# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))

# opennre_model = opennre.get_model('entity')
# opennre_model = opennre_model.to('cuda:1')


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        # features= features[:2]
        text = [i['sentence'] for i in features]
        types = [i['type'] for i in features]
        labels = [i['labels'] for i in features]
        id = [i['id'] for i in features]

        sketch = []
        n_text = []

        # start = time.process_time()

        for i in range(len(text)): # for ever datapoint in a batch
            # print('Text: ', text[i])
            new_text, new_type, new_label = text[i], types[i], labels[i]
            assert len(new_text) == len(new_type) == len(new_label)
            original_text = ' '.join(copy.deepcopy(new_text))
            # print('Original Text: ', original_text)
            linearize(new_text, new_label, args.shouldLinearizeAllWords)
            final_y = ' '.join(copy.deepcopy(new_text))
            mask_spacy_entities(new_text, new_type, args.mean, args.std)
            generated_sketch = add_relations(new_text, original_text, id[i], train_precompute, dev_precompute)
            # generated_sketch = merge_list(new_text)
            # print('Sketch: ', generated_sketch)
            sketch.append(generated_sketch)
            n_text.append(final_y)
            # print('Pretrain Text Y: ', final_y, '\n\n')
            # n_text.append(copy_text)
            # sketch.append(new_sketch)
            # print('Text: ', copy_text)
            # print('Sketch: ', new_sketch, '\n\n')

        # print(sketch,'\n',n_text)
        # fdafddfa
        # print(time.process_time() - start)

        model_inputs = tokenizer(sketch, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(n_text, max_length=max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']

        features = []
        for i in range(len(model_inputs['labels'])):
            features.append({'input_ids': model_inputs['input_ids'][i],
                            'attention_mask': model_inputs['attention_mask'][i],
                            'labels': model_inputs['labels'][i] })

        del model_inputs, labels, sketch, n_text, text, types

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

# ROUGE metric：
# rouge_score = load_metric("rouge")
def compute_metrics(eval_pred):
    # predictions, labels = eval_pred
    # # Decode generated summaries into text
    # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # # Replace -100 in the labels as we can't decode them
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # # Decode reference summaries into text
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # # ROUGE expects a newline after each sentence
    # decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # # Compute ROUGE scores
    # result = rouge_score.compute(
    #     predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    # )
    # # Extract the median scores
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {}


##################################################################
#                     training
##################################################################

batch_size = args.batch_size
num_train_epochs = args.epochs
model_name = model_checkpoint.split("/")[-1]

# load the pretrained weights
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# load only the model, without weights
# config = AutoConfig.from_pretrained(model_checkpoint)
# model =  AutoModel.from_config(config)

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))

# new_tokens = ['<b-gene>', '<i-gene>', '<b-disease>', '<i-disease>', '<b-chemical>', '<i-chemical>', '<b-i>', '<i-i>', '<b-o>', '<i-o>', '<b-p>', '<i-p>', '<b-protein>', '<i-protein>', '<b-dna>', '<i-dna>', '<b-rna>', '<i-rna>', '<b-cell_type>', '<i-cell_type>']


with torch.no_grad():
    if 'bc2gm' in args.directory:
        # new_tokens = ['<b-gene>', '<i-gene>']
        model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[51738, :]
        model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[51738, :]
    elif 'bc5dr' in args.directory:
        # new_tokens = ['<b-disease>', '<i-disease>', '<b-chemical>', '<i-chemical>']
        model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[22987, :]
        model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[22987, :]
        model.model.encoder.embed_tokens.weight[-3, :] += model.model.encoder.embed_tokens.weight[58994, :]
        model.model.encoder.embed_tokens.weight[-4, :] += model.model.encoder.embed_tokens.weight[58994, :]
    elif 'ebmnlp' in args.directory:
        # new_tokens = ['<b-i>', '<i-i>', '<b-out>', '<i-out>', '<b-p>', '<i-p>']
        model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[81674, :]
        model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[81674, :]
        model.model.encoder.embed_tokens.weight[-3, :] += model.model.encoder.embed_tokens.weight[69933, :]
        model.model.encoder.embed_tokens.weight[-4, :] += model.model.encoder.embed_tokens.weight[69933, :]
        model.model.encoder.embed_tokens.weight[-5, :] += model.model.encoder.embed_tokens.weight[52761, :]
        model.model.encoder.embed_tokens.weight[-6, :] += model.model.encoder.embed_tokens.weight[52761, :]
    elif 'jnlpba' in args.directory:
        # new_tokens = ['<b-protein>', '<i-protein>', '<b-dna>', '<i-dna>', '<b-rna>', '<i-rna>', '<b-cell_type>', '<i-cell_type>']
        model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[7841, :]
        model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[7841, :]
        model.model.encoder.embed_tokens.weight[-3, :] += model.model.encoder.embed_tokens.weight[54674, :]
        model.model.encoder.embed_tokens.weight[-4, :] += model.model.encoder.embed_tokens.weight[54674, :]
        model.model.encoder.embed_tokens.weight[-5, :] += model.model.encoder.embed_tokens.weight[54674, :]
        model.model.encoder.embed_tokens.weight[-6, :] += model.model.encoder.embed_tokens.weight[54674, :]
        model.model.encoder.embed_tokens.weight[-7, :] += model.model.encoder.embed_tokens.weight[43092, :]
        model.model.encoder.embed_tokens.weight[-8, :] += model.model.encoder.embed_tokens.weight[43092, :]
    elif 'ncbi' in args.directory:
        # new_tokens = ['<b-disease>', '<i-disease>']
        model.model.encoder.embed_tokens.weight[-1, :] += model.model.encoder.embed_tokens.weight[58994, :]
        model.model.encoder.embed_tokens.weight[-2, :] += model.model.encoder.embed_tokens.weight[58994, :]


# logging_steps = len(tokenized_dataset['train']) // batch_size
if args.directory[-1]!='/':
    args.directory += '/'

output_dir = f"{args.directory}{args.train_file}-{args.file_name}"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy = 'epoch',
    save_total_limit = 1,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    fp16 = True,
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=60,
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# remove unnecessary columns
# tokenized_dataset["train"] = tokenized_dataset["train"].remove_columns(['bert_att', 'label'])
# tokenized_dataset["validation"] = tokenized_dataset["validation"].remove_columns(['bert_att', 'label'])

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()
# asdasd
save_path = output_dir+"-final"
trainer.save_model(save_path)

shutil.rmtree(output_dir)