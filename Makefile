LANGUAGE := id
N_CHARS_FILESYSTEM := 2
MAX_LINES := 1000000

N_CHARS_FILESYSTEM := $(if $(filter-out $(LANGUAGE), ar),$(N_CHARS_FILESYSTEM),3)

DATA_DIR_BASE := ./data
DATA_DIR := $(DATA_DIR_BASE)/$(LANGUAGE)
EMBEDDINGS_DIR_BASE := ./checkpoints
EMBEDDINGS_DIR := $(CHECKPOINTS_DIR_BASE)/$(LANGUAGE)/

WIKI_RAW_FILE := $(DATA_DIR)/wiki.txt
WIKI_SHUFFLED_FILE := $(DATA_DIR)/shuffled.txt
WIKI_WORDS_FILE := $(DATA_DIR)/tgt_words.pickle

EMB_DIR_BASE := embeddings/
EMB_DIR := $(EMB_DIR_BASE)/$(LANGUAGE)/
EMB_RAW_DIR := $(EMB_DIR)sentences/
EMB_RAW_DONE := $(EMB_DIR)done.txt
# EMB_MERG_DIR := $(EMB_DIR)merged/


full: get_embeddings
	echo "Finished training" $(LANGUAGE)

# merge_embeddings: $(EMB_VAR_FILE)
# 	echo "Merged embeddings" $(LANGUAGE)

get_embeddings: $(EMB_RAW_DONE)
	echo "Got embeddings" $(LANGUAGE)

get_data: $(WIKI_WORDS_FILE)
	echo "Got data" $(LANGUAGE)

clean:
	rm -rf $(CHECKPOINTS_DIR)
	rm $(WIKI_SHUFFLED_FILE)
	rm $(WIKI_WORDS_FILE)

# $(EMB_MERG_DIR):
# 	echo 'Merge embeddings' $(EMB_MERG_DIR)
# 	mkdir -p $(EMB_MERG_DIR)
# 	python src/h02_bert_embeddings/merge_embeddings_per_word.py --dump-size 20 --n-chars-filesystem $(N_CHARS_FILESYSTEM) \
# 		--embeddings-raw-path $(EMB_RAW_DIR)  --embeddings-merged-path $(EMB_MERG_DIR)

# Get Bert embeddings per word
$(EMB_RAW_DONE): | $(WIKI_SHUFFLED_FILE)
	mkdir -p $(EMB_RAW_DIR)
	python src/get_bert_embeddings.py --dump-size 5000 --batch-size 128 \
		--wikipedia-tokenized-file $(WIKI_SHUFFLED_FILE) --embeddings-raw-path $(EMB_RAW_DIR)
	touch $(EMB_RAW_DONE)

# $(WIKI_WORDS_FILE): | $(WIKI_SHUFFLED_FILE)
# 	python src/h01_data/get_target_words.py --language $(LANGUAGE) \
# 		--wikipedia-tokenized-file $(WIKI_SHUFFLED_FILE) --wikipedia-train-file $(WIKI_TRAIN_FILE) \
# 		--wikipedia-words-file $(WIKI_WORDS_FILE)

# Shuffle Data
$(WIKI_SHUFFLED_FILE): | $(WIKI_RAW_FILE)
	shuf $(WIKI_RAW_FILE) -n $(MAX_LINES) -o $(WIKI_SHUFFLED_FILE)
