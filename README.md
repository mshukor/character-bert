# CharacterBERT for general domain downstream tasks

[paper]: https://arxiv.org/abs/2010.10392


This repository contains an implementation of several experiments aiming at investigating the effectiveness of [CharacterBERT][paper] and compare it to BERT for some downstream tasks. The work is done during the Speech and NLP course at the MVA master at ENS Paris-Saclay.

## Abstract
It is known that Wordpieces embedding, which is used in BERT variants, induces a bias when a model is trained on a general domain and fine tuned on a specific domain. To overcome this, CharacterBERT is a new variant that replaces the wordpieces embedding by the ELMo's Character-CNN module, that instead, consults each character. We will investigate the effectiveness of this new variant and compare it to BERT on several downstream NLP tasks. As the authors showed encouraging results on downstream medical domain tasks, as well as for robustness against noise and misspellings, in this work we will address another question and show that even for general domain downstream tasks, CharacterBERT is slightly better than BERT, but at the expense of loosing training speed.

## Introduction
Language models pretraining have achieved tremendous success in downstream natural language processing (NLP) tasks, which is due to the good language representation obtained by such models. 
The best techniques so far are based on bidirectional models (such as BERT) with wordpieces tokenization, which outperformed the previous unidirectional (left to right and right to left) models in learning language representations. As a result, many variants of BERT were adopted and used in the recent years enjoying the expressive and modeling power of BERT. 

Due the growing complexity of such models, the standard way to train a model on a specific language domain is by fine-tuning a model trained on a general domain. For wordpieces based models, how effective the transfer is depends on how suitable the general domain vocabulary is to the specific domain one, a constraint that, if discarded, introduces a bias in the transfer process. 

A recent variant (CharacterBERT) proposes to produce a word level contextual representation by consulting each character, which avoid any bias. Specifically, BERT's wordpieces embedding layer is replaced by the ELMo's Character-CNN module while keeping the remaining architecture, which is also more robust to noise and misspellings. 

The aim of this project is to compare BERT and CharachterBERT and test the effectiveness of the latter on some downstream NLP tasks (such as classification and sequence labeling). 
The authors showed that CharacterBERT is better than BERT when finetuning it on a specific domain. In addition they showed that it is also robust to noise and misspelling. 
In this work, we are curious about answering the following question: Is the Character-CNN layer used in CharacterBERT better than the embedding layer used in BERT for general domain downstream tasks? that is, after excluding any source of bias, noise and misspelling, is there any benefits of using CharacterBERT?\\
As CharacterBERT is the same as BERT except for the embedding layer, we will start by explaining BERT before explaining the working principle of the Character-CNN. Finally we will present the experiments that we did to answer the underlying question.

<div style="text-align:center">
    <br>
    <img src="./img/archi-compare.png" width="45%"/>
</div>

### Installation

We recommend using a virtual environment that is specific to using CharacterBERT.

If you do not already have `conda` installed, you can install Miniconda from [this link](https://docs.conda.io/en/latest/miniconda.html#linux-installers) (~450Mb). Then, check that conda is up to date:

```bash
conda update -n base -c defaults conda
```

And create a fresh conda environment (~220Mb):

```bash
conda create python=3.8 --name=character-bert
```

If not already activated, activate the new conda environment using:

```bash
conda activate character-bert
```

Then install the following packages (~3Gb):

```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
pip install transformers==3.3.1 scikit-learn==0.23.2
```

> Note 1: If you will not be running experiments on a GPU, install pyTorch via this command instead `conda install pytorch cpuonly -c pytorch`

> Note 2: If you just want to be able to load pre-trained CharacterBERT weigths, you do not have to install `scikit-learn` which is only used for computing Precision, Recall, F1 metrics during evaluation.

### Pre-trained models

You can use the `download.py` script to download any of the models below:

| Keyword                | Model description                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| general_character_bert | General Domain CharacterBERT pre-trained from scratch on English Wikipedia and [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/).                                                                                                                            |
| medical_character_bert | Medical Domain CharacterBERT initialized from **general_character_bert** then further pre-trained on [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) clinical notes and [PMC OA](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) biomedical paper abstracts. |
| general_bert           | General Domain BERT pre-trained from scratch on English Wikipedia and [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/). <sup>1</sup>                                                                                                                        |
| medical_bert           | Medical Domain BERT initialized from **general_bert** then further pre-trained on [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) clinical notes and [PMC OA](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) biomedical paper abstracts. <sup>2</sup>       |
| bert-base-uncased      | The original General Domain [BERT (base, uncased)](https://github.com/google-research/bert#pre-trained-models)                                                                                                                                                                                                                          |

> <sup>1, 2</sup> <small>We pre-train BERT models as well so that we can fairly compare each CharacterBERT model to it's BERT counterpart. Our BERT models use the same architecture and vocabulary as `bert-base-uncased`.</small><br>

For example, to download the medical version of CharacterBERT you can run:

```bash
python download.py --model='medical_character_bert'
```

Or you can download all models by running:

```bash
python download.py --model='all'
```

### Classification task (Sentiment analysis)
This task consists of classifying movie reviews (IMDB datset) as positive or negative.

To download and unzip the dataset:
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar zxvf aclImdb_v1.tar.gz
mv aclImdb/ data/classification/imdb
```

Then you should preprocess the dataset:

```
python preprocess_imdb.py --datadir data/classification/imdb/
```

To train general CharacterBERT:
```
python main.py \
    --task='classification' \
    --embedding='general_character_bert' \
    --do_lower_case \
    --do_predict \
    --dataset-name "imdb" \
    --num_train_epochs 10 \
    --train_batch_size 6 \
    --eval_batch_size 6 \
    --validation_ratio 0.15 \
    --train_size 100 \
    --do_train \
```
To train BERT:
```
python main.py \
    --task='classification' \
    --embedding='general_bert' \
    --do_lower_case \
    --do_predict \
    --dataset-name "imdb" \
    --num_train_epochs 10 \
    --train_batch_size 6 \
    --eval_batch_size 6 \
    --validation_ratio 0.15 \
    --train_size 100 \
    --do_train \
```
CharacterBERT should give slightly better preformance than BERT.

### Sequence Labeling task (Intent Recognition)
Text intent recognition is the process of understanding a user's end goal given what they have said or typed. Intent recognition is the first step in turning a human request into a machine-executable command. 

First, you should download data (Snips Dataset or Atis dataset) and put it in the data directory.

Then you should preprocess the dataset:

```
python preprocessing.py --data data/
```

To train BERT and CharacterBERT models, you can use the same commands as in Classification task except by modifying the task parameter to sequence_labeling.
Also, you should obtain slightly similar results for BERT and CharacterBERT.

### Using CharacterBERT in practice

CharacterBERT's architecture is almost identical to BERT, so you can easilly adapt any code that uses the [Transformers](https://github.com/huggingface/transformers) library.

#### Example 1: getting word embeddings from CharacterBERT

```python
"""Basic example: getting word embeddings from CharacterBERT"""
from transformers import BertTokenizer
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer

# Example text
x = "Hello World!"

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained(
    './pretrained-models/bert-base-uncased/')
x = tokenizer.basic_tokenizer.tokenize(x)

# Add [CLS] and [SEP]
x = ['[CLS]', *x, '[SEP]']

# Convert token sequence into character indices
indexer = CharacterIndexer()
batch = [x]  # This is a batch with a single token sequence x
batch_ids = indexer.as_padded_tensor(batch)

# Load some pre-trained CharacterBERT
model = CharacterBertModel.from_pretrained(
    './pretrained-models/medical_character_bert/')

# Feed batch to CharacterBERT & get the embeddings
embeddings_for_batch, _ = model(batch_ids)
embeddings_for_x = embeddings_for_batch[0]
print('These are the embeddings produces by CharacterBERT (last transformer layer)')
for token, embedding in zip(x, embeddings_for_x):
    print(token, embedding)
```

#### Example 2: using CharacterBERT for binary classification 

```python
""" Basic example: using CharacterBERT for binary classification """
from transformers import BertForSequenceClassification, BertConfig
from modeling.character_bert import CharacterBertModel

#### LOADING BERT FOR CLASSIFICATION ####

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)  # binary classification
model = BertForSequenceClassification(config=config)

model.bert.embeddings.word_embeddings  # wordpiece embeddings
>>> Embedding(30522, 768, padding_idx=0)

#### REPLACING BERT WITH CHARACTER_BERT ####

character_bert_model = CharacterBertModel.from_pretrained(
    './pretrained-models/medical_character_bert/')
model.bert = character_bert_model

model.bert.embeddings.word_embeddings  # wordpieces are replaced with a CharacterCNN
>>> CharacterCNN(
        (char_conv_0): Conv1d(16, 32, kernel_size=(1,), stride=(1,))
        (char_conv_1): Conv1d(16, 32, kernel_size=(2,), stride=(1,))
        (char_conv_2): Conv1d(16, 64, kernel_size=(3,), stride=(1,))
        (char_conv_3): Conv1d(16, 128, kernel_size=(4,), stride=(1,))
        (char_conv_4): Conv1d(16, 256, kernel_size=(5,), stride=(1,))
        (char_conv_5): Conv1d(16, 512, kernel_size=(6,), stride=(1,))
        (char_conv_6): Conv1d(16, 1024, kernel_size=(7,), stride=(1,))
        (_highways): Highway(
        (_layers): ModuleList(
            (0): Linear(in_features=2048, out_features=4096, bias=True)
            (1): Linear(in_features=2048, out_features=4096, bias=True)
        )
        )
        (_projection): Linear(in_features=2048, out_features=768, bias=True)
    )

#### PREPARING RAW TEXT ####

from transformers import BertTokenizer
from utils.character_cnn import CharacterIndexer

text = "CharacterBERT attends to each token's characters"
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = bert_tokenizer.basic_tokenizer.tokenize(text) # this is NOT wordpiece tokenization

tokenized_text
>>> ['characterbert', 'attends', 'to', 'each', 'token', "'", 's', 'characters']

indexer = CharacterIndexer()  # This converts each token into a list of character indices
input_tensor = indexer.as_padded_tensor([tokenized_text])  # we build a batch of only one sequence
input_tensor.shape
>>> torch.Size([1, 8, 50])  # (batch_size, sequence_length, character_embedding_dim)

#### USING CHARACTER_BERT FOR INFERENCE ####

output = model(input_tensor)[0]
>>> tensor([[-0.3378, -0.2772]], grad_fn=<AddmmBackward>)  # class logits
```

For more complete (but still illustrative) examples you can refer to the `run_experiments.sh` script which runs a few Classification/SequenceLabelling experiments using BERT/CharacterBERT.

```bash
bash run_experiments.sh
```

You can adapt the `run_experiments.sh` script to try out any available model. You should also be able to add real classification and sequence labelling tasks by adapting the `data.py` script.


## Aknowledgement

The implementation was mainly based on the author original [implementation](https://github.com/helboukkouri/character-bert)
