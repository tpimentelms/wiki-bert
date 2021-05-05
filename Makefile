LANGUAGE := id
MAX_LINES := 1000000

DATA_DIR_BASE := ./data
DATA_DIR := $(DATA_DIR_BASE)/$(LANGUAGE)
WIKI_RAW_FILE := $(DATA_DIR)/wiki.txt
WIKI_SHUFFLED_FILE := $(DATA_DIR)/shuffled.txt

EMB_DIR_BASE := embeddings/
EMB_DIR := $(EMB_DIR_BASE)/$(LANGUAGE)/
EMB_RAW_DIR := $(EMB_DIR)sentences/
EMB_RAW_DONE := $(EMB_DIR)done.txt


full: get_embeddings
	echo "Finished training" $(LANGUAGE)

get_embeddings: $(EMB_RAW_DONE)
	echo "Got embeddings" $(LANGUAGE)

clean:
	rm -rf $(WIKI_SHUFFLED_FILE)
	rm -rf $(EMB_DIR)

# Get Bert embeddings per word
$(EMB_RAW_DONE): | $(WIKI_SHUFFLED_FILE)
	mkdir -p $(EMB_RAW_DIR)
	python src/get_bert_embeddings.py --dump-size 5000 --batch-size 128 \
		--wikipedia-tokenized-file $(WIKI_SHUFFLED_FILE) --embeddings-raw-path $(EMB_RAW_DIR)
	touch $(EMB_RAW_DONE)

# Shuffle Data
$(WIKI_SHUFFLED_FILE): | $(WIKI_RAW_FILE)
	shuf $(WIKI_RAW_FILE) -n $(MAX_LINES) -o $(WIKI_SHUFFLED_FILE)
