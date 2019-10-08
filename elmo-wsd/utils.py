from collections import defaultdict
from itertools import chain
from typing import List, Set, Dict

import gensim
import matplotlib.pyplot as plt
import numpy as np

import constants as const


def read_dataset(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def write_dataset(filename: str, lines: List):
    """
    Writes a list of string in a file.
    :param filename: path where to save the file.
    :param lines: list of strings to serilize.
    :return:
    """
    with open(filename, "w", encoding="utf8") as file:
        file.writelines(line + "\n" for line in lines)


def merge_txt_files(input_files: List[str], output_filename: str):
    """
    Merge the given text files.
    :param input_files: list of strings.
    :param output_filename: filename of the output file
    """
    with open(output_filename, "w", encoding="utf8") as out_file:
        out_file.writelines(line + "\n" for line in input_files)


def read_dictionary(filename: str) -> Dict:
    """
    Open a dictionary from file, in the format key -> value
    :param filename: file to read.
    :return: a dictionary.
    """
    dictionary = defaultdict(list)
    with open(filename) as file:
        for l in file:
            k, *v = l.split()
            dictionary[k] += v
    return dictionary


def write_dictionary(filename: str, dictionary: Dict):
    """
    Writes a dictionary as a file.
    :param filename: file where to save the dictionary.
    :param dictionary: dictionary to serialize.
    :return:
    """
    with open(filename, mode="w") as file:
        for k, *v in dictionary.items():
            file.write(k + " " + " ".join(v) + "\n")


def write_word_index(filename: str, dictionary: Dict):
    """
    Writes a dictionary as a file.
    :param filename: file where to save the dictionary.
    :param dictionary: dictionary to serialize.
    :return:
    """
    with open(filename, mode="w") as file:
        for k, v in dictionary.items():
            file.write(k + " " + str(v) + "\n")


def compute_word_sysnet_map(paths: List) -> Dict[str, Set]:
    """
    Produce a dictionary word -> synsets.
    :param paths: path of the input file.
    :return: a dictionary of word and synsets.
    """
    word_synset_map = defaultdict(set)
    for path in paths:
        with open(path) as file:
            # flat list of words
            words = chain.from_iterable(line.strip().split() for line in file)
            # filter senses from words
            senses = (s.lower().rpartition("_") for s in words if "_wn:" in s)
            for lemma, _, synset in senses:
                # if synset in mapping:
                word_synset_map[lemma].add(synset)
    return word_synset_map


def w2v_txt_to_bin(path_input: str, path_output: str):
    """
    Convert txt embeddings to binary.
    :param path_input: embeddings path.
    :param path_output: output path.
    :return:
    """
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path_input, binary=False)
    w2v.save_word2vec_format(path_output, binary=True)


def restrict_w2v(w2v, restricted_word_set):
    """
    Retrain from w2v model only words in the restricted word set.
    :param w2v: word2vec model
    :param restricted_word_set:v set of words to keep.
    :return: word2vec model restricted.
    """
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        vec_norm = w2v.vectors_norm[i] if w2v.vectors_norm else []
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            if vec_norm:
                new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    if new_vectors_norm:
        w2v.vectors_norm = np.array(new_vectors_norm)
    return w2v


def load_mode_map(mode: str) -> Dict:
    """
    Load inventory mapping based on mode value.
    :param mode: type of synset:
        bn: BabelNet synset
        dom: WordNet domains
        lex: LexNames labels
    :return: the inventory dictionary.
    """
    if mode == "dom":
        return read_dictionary(const.BN2DOM_MAP)
    if mode == "lex":
        return read_dictionary(const.BN2LEX_MAP)
    else:
        return {}


def timer(start: float, end: float) -> str:
    """
    Timer function. Compute execution time from strart to end (end - start).
    :param start: start time
    :param end: end time
    :return: end - start
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def plot(history):
    """
    Plot validation and training accuracy, validation and training loss over time.
    """
    fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(history.history["loss"])
    axes[0].plot(history.history["val_loss"])
    axes[0].legend(
        ["loss", "val_loss"],
        loc="upper right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    axes[1].set_ylabel("Bn Loss", fontsize=14)
    # axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(history.history["bn_loss"])
    axes[1].plot(history.history["val_bn_loss"])
    axes[1].legend(
        ["bn_loss", "val_bn_loss"],
        loc="lower right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    axes[2].set_ylabel("Dom Loss", fontsize=14)
    axes[2].set_xlabel("Epoch", fontsize=14)
    axes[2].plot(history.history["dom_loss"])
    axes[2].plot(history.history["val_dom_loss"])
    axes[2].legend(
        ["dom_loss", "val_dom_loss"],
        loc="lower right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    axes[3].set_ylabel("Lex Loss", fontsize=14)
    axes[3].set_xlabel("Epoch", fontsize=14)
    axes[3].plot(history.history["lex_loss"])
    axes[3].plot(history.history["val_lex_loss"])
    axes[3].legend(
        ["lex_loss", "val_lex_loss"],
        loc="lower right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    plt.show()
