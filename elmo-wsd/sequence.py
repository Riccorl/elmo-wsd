from typing import Dict, List

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import Sequence


class TextSequence(Sequence):
    def __init__(
        self,
        features,
        labels,
        word_index,
        batch_size: int = 64,
        max_len: int = 20,
        pad: bool = True,
        elmo: bool = False,
    ):
        self.x, self.y = features, labels
        self.word_index = word_index
        self.batch_size = batch_size
        self.max_len = max_len
        self.pad = pad
        self.elmo = elmo

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        max_batch_len = len(max(self.x[start:end], key=len))

        # truncate the sequence
        if self.max_len > 0:
            max_batch_len = (
                max_batch_len if max_batch_len < self.max_len else self.max_len
            )

        if self.elmo:
            batch_x = self.compute_x_elmo(self.x[start:end], max_batch_len, self.pad)
        else:
            batch_x = self.compute_x(
                self.x[start:end], self.word_index, max_batch_len, self.pad
            )
        batch_y = self.compute_x(
            self.y[start:end], self.word_index, max_batch_len, self.pad
        )
        return batch_x, np.expand_dims(batch_y, -1)

    @staticmethod
    def compute_x_elmo(features, max_len: int = 200, pad: bool = True) -> np.ndarray:
        """
        Compute the features X for Elmo embeddings.
        :param features: feature file.
        :param max_len: max len to pad.
        :param pad: If True pad the matrix, otherwise return the matrix not padded.
        :return: the feature vectors.
        """
        if pad:
            return np.array(
                [TextSequence._trunc_string(f, max_len) for f in features if f]
            )
        else:
            return np.array([" ".join(s) for s in features])

    @staticmethod
    def _trunc_string(text: List[str], max_len: int) -> str:
        """
        Truncate a string if len > max_len.
        :param text: string to truncate.
        :param max_len: maximum length of a string.
        :return: the string truncated
        """
        text = text if len(text) < max_len else text[:max_len]
        return " ".join(text)

    @staticmethod
    def compute_x(
        features, vocab: Dict[str, int], max_len: int = 200, pad: bool = True
    ) -> np.ndarray:
        """
        Compute the features X.
        :param features: feature file.
        :param vocab: vocab.
        :param max_len: max len to pad.
        :param pad: If True pad the matrix, otherwise return the matrix not padded.
        :return: the feature vectors.
        """
        data = [
            [vocab[word] if word in vocab else vocab["<UNK>"] for word in l]
            for l in features
            if l
        ]
        if pad:
            return pad_sequences(
                data, truncating="post", padding="post", maxlen=max_len
            )
        else:
            return np.array(data)


class MultitaskSequence(TextSequence):
    def __init__(
        self,
        features,
        pos,
        labels_bn,
        labels_dom,
        labels_lex,
        word_index_bn,
        word_index_dom,
        word_index_lex,
        word_index_pos=None,
        batch_size: int = 64,
        max_len: int = 20,
        pad: bool = True,
        elmo: bool = False,
    ):
        self.pos = pos
        self.y_dom = labels_dom
        self.y_lex = labels_lex
        self.word_index_dom = word_index_dom
        self.word_index_lex = word_index_lex
        self.word_index_pos = word_index_pos
        super().__init__(
            features, labels_bn, word_index_bn, batch_size, max_len, pad, elmo
        )

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        max_batch_len = len(max(self.x[start:end], key=len))

        # truncate the sequence
        if self.max_len > 0:
            max_batch_len = (
                max_batch_len if max_batch_len < self.max_len else self.max_len
            )

        if self.elmo:
            batch_x = self.compute_x_elmo(self.x[start:end], max_batch_len, self.pad)
            if self.word_index_pos:
                pos_x = self.compute_x(
                    self.pos[start:end], self.word_index_pos, max_batch_len, self.pad
                )
                batch_x = [batch_x, pos_x]
        else:
            batch_x = self.compute_x(
                self.x[start:end], self.word_index, max_batch_len, self.pad
            )

        batch_y_bn = self.compute_x(
            self.y[start:end], self.word_index, max_batch_len, self.pad
        )
        batch_y_dom = self.compute_x(
            self.y_dom[start:end], self.word_index_dom, max_batch_len, self.pad
        )
        batch_y_lex = self.compute_x(
            self.y_lex[start:end], self.word_index_lex, max_batch_len, self.pad
        )
        batch_y_bn = np.expand_dims(batch_y_bn, -1)
        batch_y_dom = np.expand_dims(batch_y_dom, -1)
        batch_y_lex = np.expand_dims(batch_y_lex, -1)
        return batch_x, [batch_y_bn, batch_y_dom, batch_y_lex]
