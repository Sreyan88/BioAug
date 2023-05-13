# BioAug
Implementation of BioAug: Conditional Generation based Data Augmentation for Low-Resource Biomedical NER

![Methodology](./assets/bioaug.jpeg)

<b> Dependencies and Setup </b>

1. Installing dependencies

        pip install -r requirements.txt

2. Download the UMLS dataset [here]()

3. Pretrain [michiyasunaga/BioLinkBERT-large](https://huggingface.co/michiyasunaga/BioLinkBERT-large) using [OpenNRE](https://github.com/thunlp/OpenNRE) \
<b>Note:</b> OpenNRE does not support pretraining random models. We had to update their code. Check [pretrain.py](./assets/pretrain.py) \
<b>Our checkpoint</b>: https://drive.google.com/file/d/1crS7O0FZvBWTF_XZNba3Kt2nVxHao8po/view?usp=sharing

4. Identify keywords and prepare your dataset in the format given in [sample-dataset](./sample-dataset/)

        Word \tabspace Label \tabspace Word Class (NOUN, PRONOUN, ...) (Can leave empty as well) \tabspace IsKeyword? (use BIO scheme)

5. Use [preprocess.py](preprocess.py) to convert into the correct format and also precompute the relations between entities. (Do update lines 75, 99 and 118 accordingly)

6. Run [train_dynamic.sh](./script/train_dynamic.sh)

<b>Note:</b> You'll need to update dataset paths in the scripts according to your own setting.
