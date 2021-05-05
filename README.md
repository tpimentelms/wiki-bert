# wiki-bert
Code to get embeddings from wikipedia with BERT

## Install

To install dependencies run:
```bash
$ conda env create -f config/environment.yml
```

And then install the appropriate version of pytorch, transformers and download spacy dependencies:
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```

## Get wikipedia data

To get and tokenize data, use the github repository [tpimentelms/wiki-tokenizer](https://github.com/tpimentelms/wiki-tokenizer).
Then put the `parsed.txt` files in path `data/<language-name>/wiki.txt`.

# Get embeddings

Get the embeddings:
```bash
$ make get_embeddings LANGUAGE=id
```

Merge embeddings per word and get their covariances:
```bash
$ make merge_embeddings LANGUAGE=id
```

# Make script for a language

Use the Makefile to do all steps above at once!
```bash
$ make LANGUAGE=id
```
