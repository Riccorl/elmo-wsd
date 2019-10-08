from typing import Dict, List
from xml.etree import cElementTree

from nltk.corpus import wordnet as wn

import constants as const
import utils


def semcor(path_input: str, path_output: str, map_path: str, label: bool):
    """
    Preprocess SemCor dataset and writes it in a single text file.
    :param path_input: semcor path.
    :param path_output: where to save the preprocessed file.
    :param map_path: the path to the gold key dictionary.
    :param label: if True, parse the semcor set as label.
    :return:
    """
    # load maps
    semcor_map = utils.read_dictionary(map_path)
    with open(path_output, mode="w", encoding="utf8") as out:
        root = cElementTree.parse(path_input).getroot()
        for sentence in root.findall(".//sentence"):
            out.write(_extract_sentence(sentence, semcor_map, label) + "\n")


def _extract_sentence(sentence, semcor_map: Dict, label: bool = False) -> str:
    """
    Extract sentence and replace words with sensekeys.
    :param sentence: sentence node of the tree.
    :return: the string with words replaced with sensekeys
    """
    text = []
    for child in sentence.findall("./*"):
        if label and child.tag == "instance":
            sense_key = semcor_map.get(child.attrib["id"])[0]
            lemma = child.attrib["lemma"].replace(" ", "_")
            synset = wn.lemma_from_key(sense_key).synset()
            wn_synset = "wn:" + str(synset.offset()).zfill(8) + synset.pos()
            synset_id = lemma + "_" + wn_synset
            text.append(synset_id)
        else:
            text.append(child.text.replace(" ", "-"))
    return " ".join(text)


def _extract_pos(sentence) -> str:
    """
    Extract sentence and replace words with sensekeys.
    :param sentence: sentence node of the tree.
    :return: the string with words replaced with sensekeys
    """
    text = []
    for child in sentence.findall("./*"):
        # if child.attrib["pos"] != ".":
        text.append(child.attrib["pos"])
    return " ".join(text)


def semcor_predict(path_input: str) -> List:
    """
    Read the data in input and returns every sentence in a list.
    :param path_input: path to semcor.
    :return: a list of sentences.
    """
    root = cElementTree.parse(path_input).getroot()
    parsed = []
    for sentence in root.findall(".//sentence"):
        text = []
        for child in sentence.findall("./*"):
            word = (
                child.attrib["id"] + ":" + child.attrib["lemma"]
                if child.tag == "instance"
                else child.text
            )
            text.append(word)
        parsed.append(" ".join(text))
    return parsed


def semcor_predict_map(path_input: str) -> List:
    """
    Read the data in input and returns every sentence in a list.
    :param path_input: path to semcor.
    :return: a list of sentences.
    """
    root = cElementTree.parse(path_input).getroot()
    parsed = []
    for sentence in root.findall(".//sentence"):
        text = []
        for child in sentence.findall("./*"):
            prop = {
                "lemma": child.attrib["lemma"].replace(" ", "_"),
                "pos": child.attrib["pos"],
            }
            if child.tag == "instance":
                prop["id"] = child.attrib["id"]
            word = {child.text.replace(" ", "_"): prop}
            text.append(word)
        parsed.append(text)
    return parsed


def convert_gold_key(path_input, path_output, mode):
    """
    Convert sensekey to bn/dom/lex id.
    :param path_input: path to the gold key file.
    :param path_output: opath where to save the converted gold key file.
    :param mode:
    :return:
    """
    key_input = utils.read_dataset(path_input)
    mode_map = utils.load_mode_map(mode)
    with open(path_output, mode="w") as file_output:
        for line in key_input:
            lemma_id, sense_key = line.split()[:2]
            if mode == "bn":
                wn_synset = wn.lemma_from_key(sense_key).synset()
                wn_synset_id = (
                    "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()
                )
                synset_id = mode_map.get(wn_synset_id)[0]
            elif mode == "dom":
                synset_id = (
                    mode_map.get(sense_key)[0]
                    if mode_map.get(sense_key)
                    else "factotum"
                )
            else:
                synset_id = mode_map.get(sense_key)[0]
            file_output.write(lemma_id + " " + synset_id + "\n")


if __name__ == "__main__":
    semcor(
        "../data/wsd_corpora/semcor/semcor.data.xml",
        "../data/train/semcor_train.txt",
        const.SEMCOR_TRAIN,
        False,
    )
    semcor(
        "../data/wsd_corpora/semcor/semcor.data.xml",
        "../data/train/semcor_label.txt",
        const.SEMCOR_TRAIN,
        True,
    )
