# wiki-bert

[![CircleCI](https://circleci.com/gh/tpimentelms/wiki-bert.svg?style=svg&circle-token=c7f8daf57faceb1397d9f5020e1bff13063da591)](https://circleci.com/gh/tpimentelms/wiki-bert)

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

Use the Makefile to get a language's embeddings:
```bash
$ make LANGUAGE=id
```
