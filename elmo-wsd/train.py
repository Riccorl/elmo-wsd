import argparse
import copy
import os
import sys
import time
from typing import List, Dict

import gensim
import tensorflow as tf

import constants as const
import models
import preprocess
import utils
from sequence import TextSequence, MultitaskSequence


def load_data():
    """
    Load data for train.
    :return: features and labels, cleaned.
    """
    print("Loading train dataset...")
    features = utils.read_dataset(const.SEMCOR_TRAIN)
    labels = utils.read_dataset(const.SEMCOR_LABEL)
    # clean senteces
    features = preprocess.clean_sentences(features)
    labels = preprocess.clean_sentences(labels)
    print("Loading dev dataset...")
    features_dev = utils.read_dataset(const.SE07_FEATURE)
    labels_dev = utils.read_dataset(const.SE07_LABEL)
    # clean senteces
    features_dev = preprocess.clean_sentences(features_dev)
    labels_dev = preprocess.clean_sentences(labels_dev)
    return features, features_dev, labels, labels_dev


def build_train_vocab(
    features: List, max_vocab: int, min_count: int, elmo: bool, path_emb: str = None
):
    """
    Build the train vocabulary.
    :param features: Features
    :param max_vocab: maximum number of unique words to consider.
    :param min_count: minimum frequency for a word to be considered.
    :param elmo: if True, it doesn't use the pre-trained embeddings
    :param path_emb: path to embeddings file.
    :return: if elmo is Tue -> returns the vocabulary
             else -> returns vocabulary and embeddings
    """
    word_counts = preprocess.get_word_counts(
        features, max_vocab=max_vocab, min_count=min_count
    )
    print("Word counts:", len(word_counts))
    if elmo:
        word_index = preprocess.build_word_index(word_counts)
        utils.write_word_index(const.VOCABS_DIR / "train_vocab_bn.txt", word_index)
        return word_index, None
    else:
        print("Loading embeddings...")
        w2v = gensim.models.KeyedVectors.load_word2vec_format(path_emb, binary=True)
        w2v = utils.restrict_w2v(w2v, set(word_counts.keys()))
        # build the sense vocab from the w2v model
        word_index = preprocess.build_word_index(w2v.index2word)
        print("Cleaned len:", len(word_index))
        return word_index, w2v


def build_label_vocab(
    labels: List, max_senses: int, min_count: int, mode: str, word_index: Dict = None
):
    """
    Build the label vocabulary.
    :param labels: Labels.
    :param word_index: the word vocabulary
    :param max_senses: maximum number of unique senses to consider.
    :param min_count: minimum frequency for a sense to be considered.
    :param mode: type of synset:
        bn: BabelNet synset
        dom: WordNet domains
        lex: LexNames labels
    :return: the label vocabulary (word vocabulary + sense vocabulary)
    """
    senses_counts = preprocess.get_word_counts(
        labels, max_vocab=max_senses, min_count=min_count, mode=mode
    )
    print("Sense counts:", len(senses_counts))
    if word_index:
        sense_index = preprocess.build_word_index(senses_counts, init=False)
        label_index = preprocess.build_word_index(
            {**word_index, **sense_index}, init=False
        )
    else:
        label_index = preprocess.build_word_index(senses_counts)
    # write dictionary
    utils.write_word_index(
        const.VOCABS_DIR / ("label_vocab_" + mode + ".txt"), label_index
    )
    return label_index


def generator(
    batch_size: int = 64,
    max_len: int = 20,
    max_vocab: int = 40000,
    min_count: int = 5,
    elmo: bool = False,
    path_emb: str = None,
):
    """
    Build the train generator for the Keras model.
    :param batch_size: size of the batch.
    :param max_vocab: maximum number of unique words to consider.
    :param max_len: maximum length of a sentence.
    :param min_count: minimum frequency for a word to be considered.
    :param elmo: True to use ELMo language model, false to use standard embeddings.
    :param path_emb: path to embeddings file.
    :return: train generator, dev genertor and the embeddings (if used)
    """
    feats, feats_dev, labels, labels_dev = load_data()

    labels_bn = preprocess.convert_synsets(copy.deepcopy(labels), mode="bn")
    labels_dev_bn = preprocess.convert_synsets(copy.deepcopy(labels_dev), mode="bn")

    # build vocabularies
    word_index, embeddings = build_train_vocab(
        feats, max_vocab, min_count, elmo, path_emb
    )
    label_index = build_label_vocab(
        labels_bn, max_vocab, min_count, mode="bn", word_index=word_index
    )

    # generators for keras
    train_gen = TextSequence(
        feats, labels_bn, label_index, batch_size, max_len, elmo=elmo
    )
    dev_gen = TextSequence(
        feats_dev, labels_dev_bn, label_index, batch_size, max_len, elmo=elmo
    )
    return train_gen, dev_gen, embeddings


def generator_multitask(
    batch_size: int = 64,
    max_len: int = 20,
    max_vocab: int = 40000,
    min_count: int = 5,
    elmo: bool = False,
    path_emb: str = None,
):
    """
    Build the train generator for the Keras multitask model.
    :param batch_size: size of the batch.
    :param max_len: maximum length of a sentence.
    :param max_vocab: maximum number of unique words to consider.
    :param min_count: minimum frequency for a word to be considered.
    :param elmo: True to use ELMo language model, false to use standard embeddings.
    :param path_emb: path to embeddings file.
    :return: train generator, dev genertor and the embeddings (if used)
    """
    feats, feats_dev, labels, labels_dev = load_data()

    # convert the labels in the corresponding ids
    labels_bn = preprocess.convert_synsets(copy.deepcopy(labels), mode="bn")
    labels_dom = preprocess.convert_synsets(copy.deepcopy(labels), mode="dom")
    labels_lex = preprocess.convert_synsets(copy.deepcopy(labels), mode="lex")
    # convert the labels in the corresponding ids
    labels_dev_bn = preprocess.convert_synsets(copy.deepcopy(labels_dev), mode="bn")
    labels_dev_dom = preprocess.convert_synsets(copy.deepcopy(labels_dev), mode="dom")
    labels_dev_lex = preprocess.convert_synsets(copy.deepcopy(labels_dev), mode="lex")

    # build vocabularies
    word_index, embeddings = build_train_vocab(feats, max_vocab, 5, elmo, path_emb)
    label_index_bn = build_label_vocab(
        labels_bn, max_vocab, min_count, mode="bn", word_index=word_index
    )
    label_index_dom = build_label_vocab(labels_dom, max_vocab, min_count, mode="dom")
    label_index_lex = build_label_vocab(labels_lex, max_vocab, min_count, mode="lex")

    # generators for keras
    train_gen = MultitaskSequence(
        feats,
        labels_bn,
        labels_dom,
        labels_lex,
        label_index_bn,
        label_index_dom,
        label_index_lex,
        batch_size,
        max_len,
        elmo=elmo,
    )
    dev_gen = MultitaskSequence(
        feats_dev,
        labels_dev_bn,
        labels_dev_dom,
        labels_dev_lex,
        label_index_bn,
        label_index_dom,
        label_index_lex,
        batch_size,
        max_len,
        elmo=elmo,
    )
    return train_gen, dev_gen, embeddings


def train(
    hidden_size: int,
    batch_size: int,
    epochs: int,
    elmo: bool,
    multitask: bool,
    output: str,
    path_emb: str = None,
):
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config_tf))

    if multitask:
        train_gen, dev_gen, embeddings = generator_multitask(
            batch_size,
            max_len=20,
            max_vocab=40000,
            min_count=5,
            elmo=elmo,
            path_emb=path_emb,
        )
        outputs_size = [
            len(train_gen.word_index),
            len(train_gen.word_index_dom),
            len(train_gen.word_index_lex),
        ]
    else:
        train_gen, dev_gen, embeddings = generator(
            batch_size,
            max_len=20,
            max_vocab=40000,
            min_count=5,
            elmo=elmo,
            path_emb=path_emb,
        )
        outputs_size = [len(train_gen.word_index)]

    model = models.keras_model(
        hidden_size=hidden_size,
        dropout=0.6,
        recurrent_dropout=0.5,
        learning_rate=0.0003,
        outputs_size=outputs_size,
        embeddings=embeddings,
        elmo=elmo,
        mtl=multitask,
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, mode="min", verbose=1, restore_best_weights=True
    )

    if not os.path.exists(str(const.MODEL_DIR)):
        os.makedirs(str(const.MODEL_DIR))
    cp_path = str(const.MODEL_DIR / "model_{epoch:02d}_{val_loss:.2f}.h5")
    cp = tf.keras.callbacks.ModelCheckpoint(
        cp_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    print("Starting training...")
    start = time.time()
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen.x) // batch_size,
        epochs=epochs,
        validation_data=dev_gen,
        validation_steps=len(dev_gen.x) // 32,
        callbacks=[cp, es],
    )
    end = time.time()
    print("Time to train: ", utils.timer(start, end))
    print("Saving model...")
    model.save_weights(output)
    print("Training complete.")

    utils.plot(history)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="path where to save the model file",
        dest="model",
        required=True,
    )
    parser.add_argument(
        "--epochs", help="number of epochs", dest="epochs", default=40, type=int
    )
    parser.add_argument(
        "--batch-size", help="size of the batch", dest="batch", default=64, type=int
    )
    parser.add_argument(
        "--units",
        help="number of hidden units in the LSTM layer",
        dest="units",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--elmo", help="use ELMo embeddings", dest="elmo", action="store_true"
    )
    parser.add_argument(
        "--mtl",
        help="use multitask-learning to predict coarse-grained synsets",
        dest="mtl",
        action="store_true",
    )
    parser.add_argument(
        "--emb", help="path to embeddings file. Required if elmo is false.", dest="emb"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.elmo and not args.emb:
        print("Missing embeddings path. Use --elmo or provide a path for word vectors.")
        sys.exit()
    train(
        hidden_size=args.units,
        batch_size=args.batch,
        epochs=args.epochs,
        elmo=args.elmo,
        multitask=args.mtl,
        output=args.model,
    )
