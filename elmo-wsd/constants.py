import pathlib

# directories
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RESOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "resources"
MODEL_DIR = RESOURCE_DIR / "checkpoints"
WSD_DIR = DATA_DIR / "wsd_corpora"
TRAIN_DIR = DATA_DIR / "train"
DEV_DIR = DATA_DIR / "dev"
MAPPING_DIR = RESOURCE_DIR / "mapping"
VOCABS_DIR = RESOURCE_DIR / "vocabs"
EMBEDDINGS_DIR = RESOURCE_DIR / "embeddings"
EVALUATION_DIR = RESOURCE_DIR / "evaluation"
PREDICTION_DIR = RESOURCE_DIR / "predicts"

# data
SEMCOR_TRAIN = TRAIN_DIR / "semcor_train.txt"
SEMCOR_POS = TRAIN_DIR / "semcor_pos.txt"
SEMCOR_LABEL = TRAIN_DIR / "semcor_label.txt"
SE07_FEATURE = DEV_DIR / "se07_features.txt"
SE07_LABEL = DEV_DIR / "se07_labels.txt"

# mapping
BN2LEX_MAP = MAPPING_DIR / "babelnet2lexnames.tsv"
BN2DOM_MAP = MAPPING_DIR / "babelnet2wndomains.tsv"
BN2WN_MAP = MAPPING_DIR / "babelnet2wordnet.tsv"
WN2BN_MAP = MAPPING_DIR / "wordnet2babelnet.txt"
LEMMA2WN_MAP = MAPPING_DIR / "lemma2wordnet.txt"
LEMMA2BN_MAP = MAPPING_DIR / "lemma2babelnet.txt"

SEMCOR_MAP = WSD_DIR / "semcor" / "semcor.gold.key.txt"
SE07MAP = WSD_DIR / "semeval2007" / "semeval2007.gold.key.txt"
OMSTI_MAP = WSD_DIR / "semcor_omsti" / "semcor+omsti.gold.key.txt"

TRAIN_VOCAB_BN = VOCABS_DIR / "train_vocab_bn.txt"
LABEL_VOCAB_BN = VOCABS_DIR / "label_vocab_bn.txt"
LABEL_VOCAB_DOM = VOCABS_DIR / "label_vocab_dom.txt"
LABEL_VOCAB_LEX = VOCABS_DIR / "label_vocab_lex.txt"

# embeddings
SENSE_VECTORS = EMBEDDINGS_DIR / "embeddings_senses.vec"
SENSE_VECTORS_BIN = EMBEDDINGS_DIR / "embeddings_senses.bin"
PRE_VECTORS = EMBEDDINGS_DIR / "embeddings.vec"
PRE_VECTORS_BIN = EMBEDDINGS_DIR / "embeddings.bin"
