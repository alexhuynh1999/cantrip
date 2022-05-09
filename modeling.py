from typing import Union, List, Callable

import tensorflow as tf

import layers
import rnn_cell

CELL_TYPES = ['LRAN', 'RAN', 'LSTM', 'GRU']


class CANTRIPModel(object):
    """reCurrent Additive Network for Temporal RIsk Predicition (CANTRIP) model

    This class contains a TensorFlow implementation of CANTRIP as described in the AMIA paper

    Attributes:
        max_seq_len (int): the maximum number of clinical snapshots used in any mini-batch
        max_snapshot_size (int):  the maximum number of observations documented in any clinical snapshot
        vocabulary_size (int): the number of unique observations
        observation_embedding_size (int): the dimensionality of embedded clinical observations
        delta_encoding_size (int): the dimensionality of delta encodings (typically 1)
        num_hidden (int or List[int]): if scalar, the number of hidden units used in the single-layer clinical picture
            inference RNN; if list, the number of hidden units used in a multi-layer stacked clinical picture inference
            RNN
        cell_type (str): the type of RNN cell to use for the clinical picture inference RNN
        batch_size (int): the size of all mini-batches
        snapshot_encoder (function): a callable function which adds clinical snapshot encoding operations to the
            TensorFlow graph; see src.models.encoder for options
        dropout (float): the dropout rate used in all dropout layers
        vocab_dropout (float): the vocabulary dropout rate (defaults to dropout)
        num_classes (int): the number of classes -- this should be two.

        observations: a tf.Tensor with shape [batch_size x max_seq_len x max_snapshot_size] and type tf.int32 containing
            the zero-padded/truncated clinical observations in each snapshot in each chronology of a single mini-batch
        deltas: a tf.Tensor with shape [batch_size x max_seq_len x delta_encoding_size] and type tf.float32 containing
            the zero-padded/truncated encoded deltas for each snapshot in each chronology of a single mini-bath
        snapshot_sizes: a tf.Tensor with shape [batch_size x max_seq_len] and type tf.int32 indicating the actual
            number of non-zero clinical observations in each clinical snapshot in each chronology of a single mini-batch
        seq_lengths: a tf.Tensor with shape [batch_size] and type.int32 indicating the original
            (pre-padding post-truncating) length of all clinical chronologies in the mini-batch
        labels: a tf.Tensor with shape [batch_size] and type.int32 indicating the label (disease-risk) that should be
            predicted for the final clinical snapshot after the final delta value

        x: a tf.Tensor with shape[batch_size x max_seq_len x (snapshot_encoding_size + delta_encoding_size) and type
            tf.float32 representing the sequential inputs to the clinical picture inference RNN
        seq_final_output: a tf.Tensor with shape [batch_size x num_hidden[-1]] indicating the final memory of the RNN
            after processing the final clinical snapshot and prediction window
        logits: a tf.Tensor with shape [batch_size x num_classes] with the raw (pre-softmax) outputs of the disease-risk
            prediction module
        y: a tf.Tensor with shape [batch_size] with the predicting disease-risk for each chronology in the mini-batch
    """

    def __init__(self,
                 max_seq_len: int,
                 max_snapshot_size: int,
                 vocabulary_size: int,
                 observation_embedding_size: int,
                 delta_encoding_size: int,
                 num_hidden: Union[int, List[int]],
                 cell_type: str,
                 batch_size: int,
                 snapshot_encoder: Callable[['CANTRIPModel'], tf.Tensor],
                 dropout: float = 0.,
                 vocab_dropout: float = None,
                 num_classes: int = 2,
                 delta_combine: str = "concat",
                 embed_delta: bool = False,
                 rnn_highway_depth: int = 3,
                 rnn_direction='forward'):
        """Initializes a new CANTRIP model with the given model parameters
        :param max_seq_len: the maximum number of clinical snapshots used in any mini-batch
        :param max_snapshot_size: the maximum number of observations documented in any clinical snapshot
        :param vocabulary_size:  the number of unique observations
        :param observation_embedding_size: the dimensionality of embedded clinical observations
        :param delta_encoding_size: the dimensionality of delta encodings (typically 1)
        :param num_hidden: if scalar, the number of hidden units used in the single-layer clinical picture
            inference RNN; if list, the number of hidden units used in a multi-layer stacked clinical picture inference
            RNN
        :param cell_type: the type of RNN cell to use for the clinical picture inference RNN
        :param batch_size: the size of all mini-batches
        :param snapshot_encoder: a callable function which adds clinical snapshot encoding operations to the
            TensorFlow graph; see src.models.encoder for options
        :param dropout: the dropout rate used in all dropout layers
        :param num_classes: num_classes (int): the number of classes -- this should be two but will work for
            multivariate (i.e., finer-grained) labels
        """
        self.max_seq_len = max_seq_len
        self.max_snapshot_size = max_snapshot_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = observation_embedding_size
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.delta_encoding_size = delta_encoding_size
        self.dropout = dropout
        self.vocab_dropout = vocab_dropout or dropout
        self.cell_type = cell_type
        self.delta_combine = delta_combine
        self.embed_delta = embed_delta
        self.rnn_highway_depth = rnn_highway_depth
        self.rnn_direction = rnn_direction

        if delta_combine == 'add' and not self.embed_delta and self.embedding_size != self.delta_encoding_size:
            print("Cannot add delta embeddings of size %d to observation encodings of size %d, "
                  "setting embed_delta=True" %
                  (self.delta_encoding_size, self.embedding_size))
            self.embed_delta = True

        # Build computation graph
        # self.regularizer = tfcontrib.layers.l1_regularizer(0.05)
        with tf.variable_scope('cantrip'):
            self._add_placeholders()
            with tf.variable_scope('snapshot_encoder'):  # , regularizer=self.regularizer):
                self.snapshot_encodings = snapshot_encoder(self)

            if self.embed_delta:
                with tf.variable_scope('delta_encoder'):
                    self.delta_inputs = tf.keras.layers.Dense(units=self.embedding_size,
                                                              activation=None,
                                                              name='delta_embeddings')(self.deltas)
            else:
                self.delta_inputs = self.deltas

            self._add_seq_rnn(cell_type)
            # Convert to sexy logits
            self.logits = tf.keras.layers.Dense(units=self.num_classes,
                                                activation=None,
                                                name='class_logits')(self.seq_final_output)
        self._add_postprocessing()

    def _add_placeholders(self):
        """Add TensorFlow placeholders/feeds which are used as inputs to the model for each mini-batch"""
        # Observation IDs
        self.observations = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len, self.max_snapshot_size],
                                           name="observations")

        # Elapsed time deltas
        self.deltas = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.delta_encoding_size],
                                     name="deltas")

        # Snapshot sizes
        self.snapshot_sizes = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name="snapshot_sizes")

        # Chronology lengths
        self.seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name="seq_lengths")

        # Label
        self.labels = tf.placeholder(tf.int32, [self.batch_size], name="labels")

        # Training
        self.training = tf.placeholder(tf.bool, name="training")

    def _add_seq_rnn(self, cell_type: str):
        """Add the clinical picture inference module; implemented in as an RNN. """
        with tf.variable_scope('sequence'):
            # Add dropout on deltas
            if self.dropout > 0:
                self.delta_inputs = tf.keras.layers.Dropout(rate=self.dropout)(self.delta_inputs,
                                                                               training=self.training)

            # Concat observation_t and delta_t (deltas are already shifted by one)
            if self.delta_combine == 'concat':
                self.x = tf.concat([self.snapshot_encodings, self.delta_inputs], axis=-1, name='rnn_input_concat')
            elif self.delta_combine == 'add':
                self.x = self.snapshot_encodings + self.delta_inputs
            else:
                raise ValueError("Invalid delta combination method: %s" % self.delta_combine)

            # Add dropout on concatenated inputs
            if self.dropout > 0:
                self.x = tf.keras.layers.Dropout(rate=self.dropout)(self.x, training=self.training)

            _cell_types = {
                # Original RAN from https://arxiv.org/abs/1705.07393
                'RAN': rnn_cell.RANCell,
                'RAN-LN': lambda units: rnn_cell.RANCell(units, normalize=True),
                'VHRAN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth),
                'VHRAN-LN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth,
                                                             normalize=True),
                'RHN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
                                                      depth=self.rnn_highway_depth,
                                                      is_training=self.training),
                'RHN-LN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
                                                         depth=self.rnn_highway_depth,
                                                         is_training=self.training,
                                                         normalize=True),
                # Super secret simplified RAN variant from Eq. group (2) in https://arxiv.org/abs/1705.07393
                # 'LRAN': lambda num_cells: rnn_cell.SimpleRANCell(self.x.shape[-1]),
                # 'LRAN-LN': lambda num_cells: rnn_cell.SimpleRANCell(self.x.shape[-1], normalize=True),
                'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
                'LSTM-LN': tf.contrib.rnn.LayerNormBasicLSTMCell,
                'GRU': tf.nn.rnn_cell.GRUCell,
                'GRU-LN': rnn_cell.LayerNormGRUCell
            }

            if cell_type not in _cell_types:
                raise ValueError('unsupported cell type %s', cell_type)

            self.cell_fn = _cell_types[cell_type]

            if self.rnn_direction == 'bidirectional':
                self.seq_final_output = layers.bidirectional_rnn_layer(self.cell_fn,
                                                                       self.num_hidden, self.x, self.seq_lengths)
            else:
                self.seq_final_output = layers.rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)

            print('Final output:', self.seq_final_output)

            # Even more fun dropout
            if self.dropout > 0:
                self.seq_final_output = \
                    tf.keras.layers.Dropout(rate=self.dropout)(self.seq_final_output, training=self.training)

    def _add_postprocessing(self):
        """Categorical arg-max prediction for disease-risk"""
        # Class labels (used mainly for metrics)
        self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')
