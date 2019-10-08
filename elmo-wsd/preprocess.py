import itertools
import re
import string
from collections import OrderedDict, Counter
from typing import List, Dict, Set, Any

import gensim
from keras_preprocessing.text import maketrans

import constants as const
import utils


def clean_sentences(
    sentences: List, filters: str = "'!\"’#$%()*+,–./<=>?@[\\]^`{}~\t\n"
) -> List[List[str]]:
    """
    Cleans the dataset in input form stopwords and punctuations.
    :param sentences: strings to clean.
    :param filters: symbols to remove.
    :return: a list of cleaned sentences.
    """
    stops = set(string.punctuation)
    html_regex = re.compile(r"&\w+;")
    translate_dict = dict((c, "") for c in filters)
    translate_map = maketrans(translate_dict)
    return [_clean_sentence(s, stops, translate_map, html_regex) for s in sentences]


def _clean_sentence(
    text: str, stops: Set, translate_map: Dict, html_regex
) -> List[str]:
    """
    Clean the string in input form stopwords and punctuations.
    :param text: string to clean.
    :param stops: stop words to remove.
    :param translate_map: symbols to remove.
    :return: the cleaned string.
    """
    text = text.translate(translate_map)
    return [
        word
        for word in text.lower().split()
        if word and word not in stops and not html_regex.search(word)
    ]
    # return [word for word in text.lower().split()]


def clean_predict(sentences: List):
    """
    Cleans the test dataset in input form stopwords and punctuations.
    :param sentences: strings to clean.
    :return: a list of cleaned sentences.
    """
    stops = set(string.punctuation)
    filters = "'!\"’#$%&()*+,–/<=>?@[\\]^`{}~\t\n"
    translate_dict = dict((c, "") for c in filters)
    translate_map = maketrans(translate_dict)
    return [_clean_predict(p, stops, translate_map) for p in sentences]


def _clean_predict(sentence, stops: Set, translate_map: Dict):
    """
    Clean the string in input form stopwords and punctuations.
    :param sentence: string to clean.
    :param stops: stop words to remove.
    :param translate_map: symbols to remove.
    :return: the cleaned string.
    """
    cleaned_sentence = []
    for word in sentence:
        w = list(word.keys())[0]
        w = w.lower()
        w = w.translate(translate_map)
        if w and w not in stops:
            word[w] = word.pop(list(word.keys())[0])
            cleaned_sentence.append(word)
    return cleaned_sentence


def build_word_counts(lines: List, mode: str = "word") -> OrderedDict:
    """
    Builds a dictionary of word frequences.
    :param lines: A list of list of strings.
    :param mode: type of synset:
        bn: BabelNet synset
        dom: WordNet domains
        lex: LexNames labels
    :return: an ordered dict of word frequences.
    """
    if mode == "bn":
        word_counts = Counter(w for l in lines for w in l if "_bn:" in w)
    elif mode == "lex" or mode == "dom":
        word_counts = Counter(
            w
            for l in lines
            for w in l
            if "UNK" not in w
        )
    else:
        word_counts = Counter(w for l in lines for w in l)
    return OrderedDict(
        sorted(word_counts.items(), key=lambda k: int(k[1]), reverse=True)
    )


def build_word_index(word_dict: Dict, init: bool = True) -> Dict:
    """
    Builds a dictionary from word to index.
    :param word_dict: Dictionary of words.
    :param init: If True, adds padding and unkown to the dictionary.
    :return: a dicitonary word -> index.
    """
    if init:
        vocab = {"<PAD>": 0, "<UNK>": 1}
        offset = 2
    else:
        vocab = {}
        offset = 0
    for i, k in enumerate(word_dict):
        if k not in vocab:
            vocab[k] = i + offset
    return vocab


def build_word_index_from_gensim(
    word2vec: gensim.models.word2vec.Word2Vec, init: bool = True
) -> Dict[str, int]:
    """
    :param word2vec: trained Gensim Word2Vec model
    :param init:
    :return: a dictionary from token to int
    """
    if init:
        vocab = {"<PAD>": 0, "<UNK>": 1}
        offset = 2
    else:
        vocab = {}
        offset = 0
    for index, word in enumerate(word2vec.wv.index2word):
        vocab[word] = index + offset
    return vocab


def get_word_counts(
    sentences: List, max_vocab: int, min_count: int, mode: str = "word"
):
    """
    Computes the word counts for each word.
    :param sentences: Sentences.
    :param max_vocab: maximum number of unique words to consider.
    :param min_count: Minimum number of occurence of a word to be considered.
    :param mode: word -> counts words
                 bn -> counts BabelNet senses
                 dom -> counts WordNet domains
                 lex -> counts LexNames tokens
    :return: a dictionary word -> number of occurence
    """
    wc_features = build_word_counts(sentences, mode=mode)
    max_vocab_counts = itertools.islice(wc_features.items(), max_vocab)
    wc_reduced = {w: c for w, c in max_vocab_counts if c >= min_count}
    return wc_reduced


def get_ids(line: List) -> Dict[Any, int]:
    """
    Get the ids from the data in input.
    :param line: a list containing words and other info (lemma, pos, id, etc.).
    :return: the line cleaned and a dictionary from ids to indices.
    """
    ids = {}
    for i, l in enumerate(line):
        if next(iter(l.items()))[1].get("id"):
            ids[next(iter(l.items()))[1].get("id")] = i
    return ids


def get_pos(line: List) -> Dict[int, str]:
    """
    Get the pos from the data in input.
    :param line: a list containing words and other info (lemma, pos, id, etc.).
    :return: a dictionary from indices to pos.
    """
    pos = {}
    for i, l in enumerate(line):
        if next(iter(l.items()))[1].get("pos"):
            pos[i] = convert_pos(next(iter(l.items()))[1].get("pos"))
    return pos


def get_lemmas(line: List) -> Dict[int, str]:
    """
    Get the lemmas from the data in input.
    :param line: a list containing words and other info (lemma, pos, id, etc.).
    :return: a dictionary from indices to lemmas.
    """
    lemmas = {}
    for i, l in enumerate(line):
        lemmas[i] = next(iter(l.items()))[1].get("lemma")
    return lemmas


def convert_pos(pos: str) -> str:
    """
    Convert pos tagging in nltk readable format.
    :param pos: pos tag to convert.
    :return: pos tag converted.
    """
    switch = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
    return switch.get(pos)


def convert_synsets(lines: List, mode: str) -> List[List[str]]:
    """
    Convert every wordnet synset in the given mode synset.
    :param lines: List of strings to convert
    :param mode: type of synset:
        bn: BabelNet synset
        dom: WordNet domains
        lex: LexNames labels
    :return:
    """
    wn2bn_map = utils.read_dictionary(const.WN2BN_MAP)
    mode_map = utils.load_mode_map(mode)

    for line in lines:
        for i, word in enumerate(line):
            if "_wn:" in word:
                lemma, _, synset = word.rpartition("_")
                line[i] = _convert_synset(lemma, synset, mode, mode_map, wn2bn_map)
            else:
                if mode == "lex" or mode == "dom":
                    line[i] = "other"
    return lines


def _convert_synset(lemma, synset, mode, mode_map, wn2bn_map):
    """
    Convert the synset to a different id.
    :param lemma: lemma of the sense.
    :param synset: synset of the sense.
    :param mode: type of synset:
        bn: BabelNet synset
        dom: WordNet domains
        lex: LexNames labels
    :param mode_map: inventory mapping based on mode value.
    :param wn2bn_map: WordNet to BabelNet synset mapping.
    :return:
        sense if the synset is in the mapping
        lemma otherwise
    """
    if wn2bn_map.get(synset):
        synset = wn2bn_map.get(synset)[0]
        if mode == "bn":
            return lemma + "_" + synset
        else:
            if mode_map.get(synset):
                return mode_map.get(synset)[0]
            else:
                return "factotum"
    return lemma
