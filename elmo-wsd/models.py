from typing import List

import gensim
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras as k
from tensorflow.python.keras import Model


def keras_model(
    hidden_size: int = 256,
    input_length: int = None,
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    learning_rate: float = None,
    vocab_size: int = None,
    embedding_size: int = 300,
    train_embeddings: bool = False,
    outputs_size: List = None,
    embeddings: gensim.models.word2vec.Word2Vec = None,
    elmo: bool = False,
    mtl: bool = False,
) -> Model:
    if elmo:
        input_layer = k.layers.Input(shape=(input_length,), dtype="string")
        em = ElmoLayer()(input_layer)
    else:
        input_layer = k.layers.Input(shape=(input_length,))
        em = _get_keras_embedding(embeddings, train_embeddings)(input_layer)

    lstm1 = k.layers.Bidirectional(
        k.layers.LSTM(
            units=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        )
    )(em)
    # Attention layer
    lstm_a = attention_layer(lstm1)
    lstm_dr = k.layers.Dropout(0.5)(lstm_a)

    if mtl:
        bn = k.layers.Dense(outputs_size[0], activation="softmax", name="bn")(lstm_dr)
        dom = k.layers.Dense(outputs_size[1], activation="softmax", name="dom")(lstm_dr)
        lex = k.layers.Dense(outputs_size[2], activation="softmax", name="lex")(lstm_dr)
        output = [bn, dom, lex]
    else:
        output = k.layers.Dense(outputs_size[0], activation="softmax")(lstm_dr)

    if learning_rate:
        optimizer = k.optimizers.Adam(lr=learning_rate)
    else:
        optimizer = k.optimizers.Adadelta()

    # Initialization elmo variables
    init_keras()
    model = k.models.Model(inputs=input_layer, outputs=output)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
    )
    model.summary()
    return model


class ElmoLayer(k.layers.Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module(
            "https://tfhub.dev/google/elmo/2",
            trainable=self.trainable,
            name="{}_module".format(self.name),
        )

        self._trainable_weights += tf.trainable_variables(
            scope="^{}_module/.*".format(self.name)
        )
        super(ElmoLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        return self.elmo(
            tf.reshape(tf.cast(inputs, tf.string), [-1]),
            as_dict=True,
            signature="default",
        )["elmo"]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.dimensions


def attention_layer(lstm):
    """
    Attention mechanism for LSTMs
    :param lstm: lstm in input
    :return: lstm concatenated with attention
    """
    # h = k.layers.Concatenate()([lstm[1], lstm[3]])
    # h = k.layers.RepeatVector(k.backend.shape(lstm[0])[1])(h)
    # u = k.layers.Dense(1, activation="tanh")(h)
    # a = k.layers.Activation("softmax")(u)
    # c = k.layers.Lambda(lambda x: k.backend.sum(x[0] * x[1], axis=1))([lstm[0], a])
    # return k.layers.Multiply()([lstm[0], c])
    h = k.layers.Concatenate()([lstm[1], lstm[3]])
    h = k.layers.RepeatVector(k.backend.shape(lstm[0])[1])(h)
    u = k.layers.Dense(1, activation="tanh")(h)
    a = k.layers.Activation("softmax")(u)
    c = k.layers.Lambda(lambda x: k.backend.sum(x[0] * x[1], axis=1))([lstm[0], a])
    return k.layers.Multiply()([lstm[0], c])


def _get_keras_embedding(
    word2vec: gensim.models.word2vec.Word2Vec, trainable: bool = False
) -> k.layers.Embedding:
    """
    Return a Tensorflow Keras 'Embedding' layer with weights set as the
    Word2Vec model's learned word embeddings.
    :param word2vec: gensim Word2Vec model
    :param trainable: If False, the weights are frozen and stopped from being updated.
                      If True, the weights can/will be further trained/updated.
    :return: a tf.keras.layers.Embedding layer.
    """
    weights = word2vec.wv.vectors
    # random vector for pad
    pad = np.random.rand(1, weights.shape[1])
    # mean vector for unknowns
    unk = np.mean(weights, axis=0, keepdims=True)
    weights = np.concatenate((pad, unk, weights))

    return k.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        mask_zero=True,
        trainable=trainable,
    )


def init_keras():
    sess = k.backend.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
