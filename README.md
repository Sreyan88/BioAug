# BioAug
Implementation of BioAug: Conditional Generation based Data Augmentation for Low-Resource Biomedical NER

![Methodology](./assets/bioaug.jpeg)

<b> Dependencies and Setup </b>

1. Installing dependencies

        pip install -r requirements.txt

2. Download the UMLS dataset [here]()

3. Pretrain [michiyasunaga/BioLinkBERT-large](https://huggingface.co/michiyasunaga/BioLinkBERT-large) using [OpenNRE](https://github.com/thunlp/OpenNRE)
Note: OpenNRE does not support pretraining random models. We had to update their code. Check [pretrain.py](./assets/pretrain.py)

4. Identify keywords and prepare your dataset in the format given in [sample-dataset](./sample-dataset/)

        Word \tabspace Label \tabspace Word Class (NOUN, PRONOUN, ...) (Can leave empty as well) \tabspace IsKeyword? (use BIO scheme)

5. Use [preprocess.py](preprocess.py) to convert into the correct format and also precompute the relations between entities. (Do update lines 75, 99 and 118 accordingly)

6. Run [train_dynamic.sh](./script/train_dynamic.sh)

Note: You'll need to update dataset paths in the scripts according to your own setting.
