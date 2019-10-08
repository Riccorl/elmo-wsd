import os
from typing import Dict, List

from nltk.corpus import wordnet as wn
from tqdm import tqdm

import models
import parse
import preprocess
import utils
from sequence import TextSequence


def predict_babelnet(input_path: str, output_path: str, resources_path: str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    _predict(input_path, output_path, resources_path, 0)
    return


def predict_wordnet_domains(
    input_path: str, output_path: str, resources_path: str
) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    _predict(input_path, output_path, resources_path, 1)
    return


def predict_lexicographer(
    input_path: str, output_path: str, resources_path: str
) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    _predict(input_path, output_path, resources_path, 2)
    return


def load_test(path_input: str) -> List:
    """
    Read and clean the test set
    :param path_input: input to test file
    :return: cleaned test set
    """
    sentences = parse.semcor_predict_map(path_input)
    return preprocess.clean_predict(sentences)


def _predict(input_path: str, output_path: str, resources_path: str, task: int = None):
    """
    Wrapper function for all the prediction functions.
    :param input_path: the path of the input file to predict in the same format as Raganato's framework.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :param task:    0 for BN
                    1 for DOM
                    2 for LEX
    :return:
    """

    print("Loading", input_path.split("/")[-1])
    sentences = load_test(input_path)

    # Loads all the mapping files.
    word_index = utils.read_dictionary(
        os.path.join(resources_path, "vocabs/label_vocab_bn.txt")
    )
    word_index_dom = utils.read_dictionary(
        os.path.join(resources_path, "vocabs/label_vocab_dom.txt")
    )
    word_index_lex = utils.read_dictionary(
        os.path.join(resources_path, "vocabs/label_vocab_lex.txt")
    )
    outputs_size = [len(word_index), len(word_index_dom), len(word_index_lex)]
    lemma2syn = utils.read_dictionary(
        os.path.join(resources_path, "mapping/lemma2wordnet.txt")
    )
    wn2bn = utils.read_dictionary(
        os.path.join(resources_path, "mapping/wordnet2babelnet.txt")
    )

    bn2coarse = None
    coarse_index = None
    if task != 0:
        # if task != 0, DOM or LEX prediction.
        coarse_index = word_index_dom if task == 1 else word_index_lex
        coarse_path = (
            os.path.join(resources_path, "mapping/babelnet2wndomains.tsv")
            if task == 1
            else os.path.join(resources_path, "mapping/babelnet2lexnames.tsv")
        )
        bn2coarse = utils.read_dictionary(coarse_path)

    print("Loading weights...")
    model = models.keras_model(
        hidden_size=256,
        dropout=0.6,
        recurrent_dropout=0.5,
        learning_rate=0.0003,
        outputs_size=outputs_size,
        elmo=True,
        mtl=True,
    )
    model.load_weights(os.path.join(resources_path, "model.h5"))

    with open(output_path, mode="w", encoding="utf8") as out_file:
        for s in tqdm(sentences):
            line = [list(l.keys())[0] for l in s]
            ids = preprocess.get_ids(s)
            pos = preprocess.get_pos(s)
            lemmas = preprocess.get_lemmas(s)
            line_input = TextSequence.compute_x_elmo([line], pad=False)
            pred = model.predict(line_input)[task]
            lables = _get_labels(
                pred,
                lemmas,
                ids,
                pos,
                lemma2syn,
                wn2bn,
                word_index,
                coarse_index,
                bn2coarse,
            )
            out_file.writelines(
                k + " " + v.rsplit("_")[-1] + "\n" for k, v in lables.items()
            )
    return


def _get_labels(
    prediction: List,
    lemmas_map: Dict,
    ids_map: Dict,
    pos_map: Dict,
    lemma2synset_map: Dict,
    wn2bn_map: Dict,
    word_index: Dict,
    coarse_index: Dict = None,
    bn2coarse: Dict = None,
):
    """
    Predict the sense for the given ids.
    :param prediction: line prediction.
    :param ids_map: dictionary from id to position inside the line.
    :param lemma2synset_map: dictionary from lemma to synsets.
    :param wn2bn_map: dictionary from WordNet to BabelNet.
    :param bn2coarse: dictionary from BabelNet to coarse.
    :param word_index: dictionary from word to index.
    :return:
    """
    pred = {}
    for sensekey, index in ids_map.items():
        lemma = lemmas_map[index]
        index_pred = prediction[0][index]
        pred[sensekey] = _get_sense(
            lemma,
            pos_map.get(index),
            lemma2synset_map,
            wn2bn_map,
            word_index,
            index_pred,
            coarse_index,
            bn2coarse,
        )
    return pred


def _get_sense(
    lemma: str,
    pos: str,
    lemma2synset_map: Dict,
    wn2bn_map: Dict,
    word_index: Dict,
    index_pred,
    coarse_index: Dict,
    bn2coarse: Dict,
):
    """
    Get the most probable sense for the given lemma id.
    :param lemma: lemma.
    :param pos:
    :param lemma2synset_map: dictionary from lemma to synsets.
    :param wn2bn_map: dictionary from WordNet to BabelNet.
    :param word_index: dictionary from word to index.
    :param index_pred: vector of proabilities for the given position.
    :return: the most probable sense.
    """
    synsets = lemma2synset_map.get(lemma)
    if synsets:
        synsets = [wn2bn_map.get(s)[0] for s in synsets if wn2bn_map.get(s)]
        if bn2coarse:
            probs = _get_probs_coarse(synsets, coarse_index, index_pred, bn2coarse)
        else:
            probs = _get_probs(synsets, lemma, word_index, index_pred)
        if probs:
            return max(probs, key=lambda p: p[1])[0]
    return get_mfs(lemma, pos, wn2bn_map, bn2coarse)


def _get_probs(synsets: List, lemma: str, word_index: Dict, index_pred):
    """
    Get the probabilites for a subset of classes
    :param synsets: list candidates synsets.
    :param lemma: lemma to predict.
    :param word_index: Word -> Index dictionary.
    :param index_pred: array of probabilites.
    :return: probabilites of synsets.
    """
    probs = []
    for s in synsets:
        sense = lemma + "_" + s
        if word_index.get(sense):
            probs.append((sense, index_pred[int(word_index.get(sense)[0])]))
    return probs


def _get_probs_coarse(synsets: List, word_index: Dict, index_pred, bn2coarse: Dict):
    """
    Get the probabilites for a subset of classes
    :param synsets: list candidates synsets.
    :param word_index: Word -> Index dictionary.
    :param index_pred: array of probabilites.
    :param bn2coarse: dictionary from BabelNet to coarse.
    :return: probabilites of synsets.
    """
    probs = []
    for s in synsets:
        if bn2coarse.get(s):
            sense = bn2coarse.get(s)[0]
            if word_index.get(sense):
                probs.append((sense, index_pred[int(word_index.get(sense)[0])]))
    return probs


def get_mfs(lemma: str, pos: str, wn2bn_map: Dict, coarse_dict=None) -> str:
    """
    Get the most frequent sense for the given lemma.
    :param lemma: lemma to get mfs.
    :param pos:
    :param wn2bn_map:
    :param coarse_dict:
    :return: the mfs.
    """
    synset = wn.synsets(lemma, pos=pos)[0]
    wn_synset = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
    pred_synset = wn2bn_map.get(wn_synset)[0]
    if coarse_dict:
        pred_synset = (
            coarse_dict.get(pred_synset)[0]
            if coarse_dict.get(pred_synset)
            else "factotum"
        )
    return lemma + "_" + pred_synset
