#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
"""
from typing import List, Tuple, Dict, Sequence, Any

from docopt import docopt
from collections import defaultdict, namedtuple
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json
import math

from chem_tensorflow import ChemModel
from utils import glorot_init, SMALL_NUMBER
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',])


class SparseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            #'batch_size': 100000,
            'batch_size': 10000,
            'use_edge_bias': False,
            'use_propagation_attention': False,
            'use_edge_msg_avg_aggregation': True,
            'residual_connections': {  # For layer i, specify list of layers whose output is added as an input
                                    "2": [0],
                                    "4": [0, 2]
                                    },
            'residual_connections': {},

            #'layer_timesteps': [2, 2, 1, 2, 1],  # number of layers & propagation steps per layer
            'layer_timesteps': [8],

            'graph_rnn_cell': 'GRU',  # GRU, CudnnCompatibleGRUCell, or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'edge_weight_dropout_keep_prob': .8
        })
        return params

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, h_dim],
                                                                          name='node_features')
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e%s' % e)
                                                for e in range(self.num_edge_types)]
        self.placeholders['num_incoming_edges_per_type'] = tf.placeholder(tf.float32, [None, self.num_edge_types],
                                                                          name='num_incoming_edges_per_type')
        self.placeholders['graph_nodes_list'] = tf.placeholder(tf.int32, [None], name='graph_nodes_list')
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')

        activation_name = self.params['graph_rnn_activation'].lower()
        if activation_name == 'tanh':
            activation_fun = tf.nn.tanh
        elif activation_name == 'relu':
            activation_fun = tf.nn.relu
        else:
            raise Exception("Unknown activation function type '%s'." % activation_name)

        # Generate per-layer values for edge weights, biases and gated units:
        self.weights = {}  # Used by super-class to place generic things
        self.gnn_weights = GGNNWeights([], [], [], [])
        for layer_idx in range(len(self.params['layer_timesteps'])):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                edge_weights = tf.Variable(glorot_init([self.num_edge_types * h_dim, h_dim]),
                                           name='gnn_edge_weights_%i' % layer_idx)
                edge_weights = tf.reshape(edge_weights, [self.num_edge_types, h_dim, h_dim])
                edge_weights = tf.nn.dropout(edge_weights, keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])
                self.gnn_weights.edge_weights.append(edge_weights)

                if self.params['use_propagation_attention']:
                    self.gnn_weights.edge_type_attention_weights.append(tf.Variable(np.ones([self.num_edge_types], dtype=np.float32),
                                                                                    name='edge_type_attention_weights_%i' % layer_idx))

                if self.params['use_edge_bias']:
                    self.gnn_weights.edge_biases.append(tf.Variable(np.zeros([self.num_edge_types, h_dim], dtype=np.float32),
                                                                    name='gnn_edge_biases_%i' % layer_idx))

                cell_type = self.params['graph_rnn_cell'].lower()
                if cell_type == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(h_dim, activation=activation_fun)
                elif cell_type == 'cudnncompatiblegrucell':
                    assert(activation_name == 'tanh')
                    import tensorflow.contrib.cudnn_rnn as cudnn_rnn
                    cell = cudnn_rnn.CudnnCompatibleGRUCell(h_dim)
                elif cell_type == 'rnn':
                    cell = tf.nn.rnn_cell.BasicRNNCell(h_dim, activation=activation_fun)
                else:
                    raise Exception("Unknown RNN cell type '%s'." % cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                     state_keep_prob=self.placeholders['graph_state_keep_prob'])
                self.gnn_weights.rnn_cells.append(cell)

    def compute_final_node_representations(self) -> tf.Tensor:
        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(self.placeholders['initial_node_representation'])
        num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int32)[0]

        message_targets = []  # list of tensors of message targets of shape [E]
        message_edge_types = []  # list of tensors of edge type of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
            edge_targets = adjacency_list_for_edge_type[:, 1]
            message_targets.append(edge_targets)
            message_edge_types.append(tf.ones_like(edge_targets, dtype=tf.int32) * edge_type_idx)
        message_targets = tf.concat(message_targets, axis=0)  # Shape [M]
        message_edge_types = tf.concat(message_edge_types, axis=0)  # Shape [M]

        for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
            with tf.variable_scope('gnn_layer_%i' % layer_idx):
                # Used shape abbreviations:
                #   V ~ number of nodes
                #   D ~ state dimension
                #   E ~ number of edges of current type
                #   M ~ number of messages (sum of all E)

                # Extract residual messages, if any:
                layer_residual_connections = self.params['residual_connections'].get(str(layer_idx))
                if layer_residual_connections is None:
                    layer_residual_states = []
                else:
                    layer_residual_states = [node_states_per_layer[residual_layer_idx]
                                             for residual_layer_idx in layer_residual_connections]

                if self.params['use_propagation_attention']:
                    message_edge_type_factors = tf.nn.embedding_lookup(params=self.gnn_weights.edge_type_attention_weights[layer_idx],
                                                                       ids=message_edge_types)  # Shape [M]

                # Record new states for this layer. Initialised to last state, but will be updated below:
                node_states_per_layer.append(node_states_per_layer[-1])
                for step in range(num_timesteps):
                    with tf.variable_scope('timestep_%i' % step):
                        messages = []  # list of tensors of messages of shape [E, D]
                        message_source_states = []  # list of tensors of edge source states of shape [E, D]

                        # Collect incoming messages per edge type
                        for edge_type_idx, adjacency_list_for_edge_type in enumerate(self.placeholders['adjacency_lists']):
                            edge_sources = adjacency_list_for_edge_type[:, 0]
                            edge_source_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                        ids=edge_sources)  # Shape [E, D]
                            all_messages_for_edge_type = tf.matmul(edge_source_states,
                                                                   self.gnn_weights.edge_weights[layer_idx][edge_type_idx])  # Shape [E, D]
                            messages.append(all_messages_for_edge_type)
                            message_source_states.append(edge_source_states)

                        messages = tf.concat(messages, axis=0)  # Shape [M, D]

                        if self.params['use_propagation_attention']:
                            message_source_states = tf.concat(message_source_states, axis=0)  # Shape [M, D]
                            message_target_states = tf.nn.embedding_lookup(params=node_states_per_layer[-1],
                                                                           ids=message_targets)  # Shape [M, D]
                            message_attention_scores = tf.einsum('mi,mi->m', message_source_states, message_target_states)  # Shape [M]
                            message_attention_scores = message_attention_scores * message_edge_type_factors

                            # The following is softmax-ing over the incoming messages per node.
                            # As the number of incoming varies, we can't just use tf.softmax. Reimplement with logsumexp trick:
                            # Step (1): Obtain shift constant as max of messages going into a node
                            message_attention_score_max_per_target = tf.unsorted_segment_max(data=message_attention_scores,
                                                                                             segment_ids=message_targets,
                                                                                             num_segments=num_nodes)  # Shape [V]
                            # Step (2): Distribute max out to the corresponding messages again, and shift scores:
                            message_attention_score_max_per_message = tf.gather(params=message_attention_score_max_per_target,
                                                                                indices=message_targets)  # Shape [M]
                            message_attention_scores -= message_attention_score_max_per_message
                            # Step (3): Exp, sum up per target, compute exp(score) / exp(sum) as attention prob:
                            message_attention_scores_exped = tf.exp(message_attention_scores)  # Shape [M]
                            message_attention_score_sum_per_target = tf.unsorted_segment_sum(data=message_attention_scores_exped,
                                                                                             segment_ids=message_targets,
                                                                                             num_segments=num_nodes)  # Shape [V]
                            message_attention_normalisation_sum_per_message = tf.gather(params=message_attention_score_sum_per_target,
                                                                                        indices=message_targets)  # Shape [M]
                            message_attention = message_attention_scores_exped / (message_attention_normalisation_sum_per_message + SMALL_NUMBER)  # Shape [M]
                            # Step (4): Weigh messages using the attention prob:
                            messages = messages * tf.expand_dims(message_attention, -1)

                        incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                                    segment_ids=message_targets,
                                                                    num_segments=num_nodes)  # Shape [V, D]

                        if self.params['use_edge_bias']:
                            incoming_messages += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
                                                           self.gnn_weights.edge_biases[layer_idx])  # Shape [V, D]

                        if self.params['use_edge_msg_avg_aggregation']:
                            num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
                                                               keep_dims=True, axis=-1)  # Shape [V, 1]
                            incoming_messages /= num_incoming_edges + SMALL_NUMBER

                        incoming_information = tf.concat(layer_residual_states + [incoming_messages],
                                                         axis=-1)  # Shape [V, D*(1 + num of residual connections)]

                        # pass updated vertex features into RNN cell
                        node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](incoming_information,
                                                                                          node_states_per_layer[-1])[1]  # Shape [V, D]

        return node_states_per_layer[-1]

    def gated_regression(self, last_h, regression_gate, regression_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]

        # Sum up all nodes per-graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 1]
        output = tf.squeeze(graph_representations)  # [g]
        self.output = output
        return output

    def classification_task(self, last_h, classification_gate, classification_transform):
        # last_h: [v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        gated_outputs = tf.nn.sigmoid(classification_gate(gate_input)) * classification_transform(last_h)  # [v x 2]

        # Sum up all nodes per-graph
        graph_representations = tf.unsorted_segment_sum(data=gated_outputs,
        #graph_representations = tf.unsorted_segment_mean(data=_output,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 2]
        output = graph_representations
        self.output = output
        self.node_outputs = gated_outputs
        return output

    def classification_task_graphb4classify(self, last_h, classification_layer):
        _input = last_h # [v x h]

        graph_representations = tf.unsorted_segment_mean(data=_input,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x h]
        output = classification_layer(graph_representations)  # [g x 2]
        self.output = output
        return output

    
    def classification_task_org(self, last_h, classification_layer):
        _input = last_h # [v x h]
        _output = classification_layer(_input)  # [v x 2]
        
        # Sum up all nodes per-graph
        #graph_representations = tf.unsorted_segment_sum(data=_output,
        graph_representations = tf.unsorted_segment_mean(data=_output,
                                                        segment_ids=self.placeholders['graph_nodes_list'],
                                                        num_segments=self.placeholders['num_graphs'])  # [g x 2]
        output = graph_representations                                                
        self.output = output
        return output


    
    def classification_task_1(self, last_h):
        # convert last hidden layer of size [v x h] to [v, 2]
        # last_h: [v x h]
        num_input_units = last_h.shape[0] * last_h.shape[1]
        num_output_units = 2
        num_hidden_units = 500

        last_h = tf.reshape(last_h, [-1, num_input_units])
        _input = last_h

        weights = {
            'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=0)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=0)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=0))
        }
        
        hidden_layer = tf.add(tf.matmul(_input, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)
        output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])

        #_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        self.output = output
        return output




    # ----- Data preprocessing and chunking into minibatches:
    #def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
    def process_raw_graphs(self, raw_data, is_training_data):
        for d in raw_data:
            (adjacency_lists, num_incoming_edge_per_type) = self.__graph_to_adjacency_lists(d['graph'])
            processed_graph  = ({"adjacency_lists": adjacency_lists,
                                 "num_incoming_edge_per_type": num_incoming_edge_per_type,
                                 "init": d["node_features"],
                                 "labels": [d["targets"][task_id][0] for task_id in self.params['task_ids']]})
            yield processed_graph

        #if is_training_data:
        #    np.random.shuffle(processed_graphs)
        #    for task_id in self.params['task_ids']:
        #        task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
        #        if task_sample_ratio is not None:
        #            ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
        #            for ex_id in range(ex_to_sample, len(processed_graphs)):
        #                processed_graphs[ex_id]['labels'][task_id] = None


    def __graph_to_adjacency_lists(self, graph) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0
            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                num_fwd_edges = self.num_edge_types/2
                assert(num_fwd_edges.is_integer())
                bwd_edge_type = int(num_fwd_edges) + edge_type   #SAHIL: bug in org code. needs div by 2 
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components."""
        #if is_training:
        #    np.random.shuffle(data)
        # Pack until we cannot fit more graphs in the batch
        state_dropout_keep_prob = self.params['graph_state_dropout_keep_prob'] if is_training else 1.
        edge_weights_dropout_keep_prob = self.params['edge_weight_dropout_keep_prob'] if is_training else 1.
        num_graphs = 0
        is_data_exhausted = False
        cur_graph = None

        try:
            while True: #num_graphs < len(data):
                num_graphs_in_batch = 0
                batch_node_features = []
                batch_target_task_values = []
                batch_target_task_mask = []
                batch_loss_weights = []
                batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
                batch_num_incoming_edges_per_type = []
                batch_graph_nodes_list = []
                node_offset = 0

                if is_data_exhausted:
                    break

                if not cur_graph:
                    cur_graph = next(data)

                #while num_graphs < len(data) and node_offset + len(data[num_graphs]['init']) < self.params['batch_size']:
                try:
                    while node_offset + len(cur_graph['init']) < self.params['batch_size']:
                        #cur_graph = data[num_graphs]
                        num_nodes_in_graph = len(cur_graph['init'])
                        padded_features = np.pad(cur_graph['init'],
                                                 ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                                 'constant')
                        batch_node_features.extend(padded_features)
                        batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph], fill_value=num_graphs_in_batch, dtype=np.int32))
                        for i in range(self.num_edge_types):
                            if i in cur_graph['adjacency_lists']:
                                batch_adjacency_lists[i].append(cur_graph['adjacency_lists'][i] + node_offset)

                        # Turn counters for incoming edges into np array:
                        num_incoming_edges_per_type = np.zeros((num_nodes_in_graph, self.num_edge_types))
                        for (e_type, num_incoming_edges_per_type_dict) in cur_graph['num_incoming_edge_per_type'].items():
                            for (node_id, edge_count) in num_incoming_edges_per_type_dict.items():
                                num_incoming_edges_per_type[node_id, e_type] = edge_count
                        batch_num_incoming_edges_per_type.append(num_incoming_edges_per_type)

                        loss_weight = 1
                        target_task_values = []
                        target_task_mask = []
                        for target_val in cur_graph['labels']:
                            if target_val is None:  # This is one of the examples we didn't sample...
                                target_task_values.append(0.)
                                target_task_mask.append(0.)
                            else:
                                target_task_values.append(target_val)
                                target_task_mask.append(1.)

                            if target_val == 1:
                                loss_weight = self.class1_weight
                            else:
                                loss_weight = self.class0_weight
                                
                        batch_loss_weights.append(loss_weight)
                        batch_target_task_values.append(target_task_values)
                        batch_target_task_mask.append(target_task_mask)
                        num_graphs += 1
                        num_graphs_in_batch += 1
                        node_offset += num_nodes_in_graph
                        cur_graph = next(data)
                except IndexError as e:
                    print(f"caught IndexError: {e}")
                    print(traceback.format_exc())
                    import pdb
                    pdb.set_trace()
                except StopIteration:
                    is_data_exhausted = True
                    #print("generator exhausted 1")

                batch_feed_dict = {
                    self.placeholders['initial_node_representation']: np.array(batch_node_features),
                    self.placeholders['num_incoming_edges_per_type']: np.concatenate(batch_num_incoming_edges_per_type, axis=0),
                    self.placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                    self.placeholders['target_values']: np.transpose(batch_target_task_values, axes=[1,0]),
                    self.placeholders['target_mask']: np.transpose(batch_target_task_mask, axes=[1, 0]),
                    self.placeholders['loss_weights']: np.array(batch_loss_weights),
                    self.placeholders['num_graphs']: num_graphs_in_batch,
                    self.placeholders['graph_state_keep_prob']: state_dropout_keep_prob,
                    self.placeholders['edge_weight_dropout_keep_prob']: edge_weights_dropout_keep_prob
                }
                #print("num_graphs_in_batch:", num_graphs_in_batch)

                # Merge adjacency lists and information about incoming nodes:
                for i in range(self.num_edge_types):
                    if len(batch_adjacency_lists[i]) > 0:
                        adj_list = np.concatenate(batch_adjacency_lists[i])
                    else:
                        adj_list = np.zeros((0, 2), dtype=np.int32)
                    batch_feed_dict[self.placeholders['adjacency_lists'][i]] = adj_list

                yield batch_feed_dict

        except StopIteration:
            #print("generator exhausted")
            pass
        #print("all done")


    def evaluate_one_batch(self, data):
        #fetch_list = self.output
        fetch_list = [self.output, tf.argmax(self.output,1)]
        #fetch_list = tf.nn.softmax(self.output)
        batch_feed_dict = self.make_minibatch_iterator(data, is_training=False)
        
        for item in batch_feed_dict:
            item[self.placeholders['graph_state_keep_prob']] = 1.0
            item[self.placeholders['edge_weight_dropout_keep_prob']] = 1.0
            item[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            item[self.placeholders['target_values']] = [[]]
            item[self.placeholders['target_mask']] = [[]]
            yield self.sess.run(fetch_list, feed_dict=item)
    
    def evaluate_batches(self, data):
        fetch_list = [self.output, tf.nn.softmax(self.output), tf.argmax(self.output,1)]
        batch_feed_dict = self.make_minibatch_iterator(data, is_training=False)
        
        y_score0 = np.empty(shape=[0,2])
        y_score1 = np.empty(shape=[0,2])
        y_pred = np.empty(shape=[0])

        for item in batch_feed_dict:
            _y_score0 = np.empty(shape=[0,2])
            _y_score1 = np.empty(shape=[0,2])
            _y_pred = np.empty(shape=[0])
            item[self.placeholders['graph_state_keep_prob']] = 1.0
            item[self.placeholders['edge_weight_dropout_keep_prob']] = 1.0
            item[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            item[self.placeholders['target_values']] = [[]]
            item[self.placeholders['target_mask']] = [[]]
            [_y_score0, _y_score1, _y_pred] = self.sess.run(fetch_list, feed_dict=item)
            y_score0 = np.concatenate((y_score0, _y_score0))
            y_score1 = np.concatenate((y_score1, _y_score1))
            y_pred = np.concatenate((y_pred, _y_pred))

        return [y_score0, y_score1, y_pred]


    def example_evaluation(self):
        ''' Demonstration of what test-time code would look like
        we query the model with the first n_example_molecules from the validation file
        '''
        n_example_molecules = 10
        with open('molecules_valid.json', 'r') as valid_file:
            example_molecules = json.load(valid_file)[:n_example_molecules]

        for mol in example_molecules:
            print(mol['targets'])

        example_molecules = self.process_raw_graphs(example_molecules, is_training_data=False)
        print(self.evaluate_one_batch(example_molecules))
   
    def _load_data(self, filename):
        fd = open(filename,'r')
        for line in fd:
            yield json.loads(line)
        fd.close()


    def print_confusion_matrix(self, y, y_pred):
        C = confusion_matrix(y,y_pred)
        tn = C[0,0]
        fn = C[1,0]
        tp = C[1,1] 
        fp = C[0,1]
        print("confusion matrix:")
        print(tp, fp)
        print(fn, tn)
        print("Accuracy:", 1.0 * (tp + tn)/(tp + tn + fp + fn))
        print("Precision:", 1.0 * tp / (tp + fp))
        print("Recall:", 1.0 * tp / (tp + fn))
    
    def print_confusion_matrix2(self, y, y_pred):
        C = confusion_matrix(y,y_pred)
        tn = C[0,0]
        fn = C[1,0]
        tp = C[1,1] 
        fp = C[0,1]
        print(tp)
        print(fn)
        print(fp)
        print(tn)
        print(1.0 * (tp + tn)/(tp + tn + fp + fn))
        print(1.0 * tp / (tp + fp))
        print(1.0 * tp / (tp + fn))



    def example_evaluation1(self):
        ''' Demonstration of what test-time code would look like
        '''
        num_samples = 1600
        num_positive_samples = 0
        samples = []
        samples_generator = self._load_data('/data/sahil/cpg_vector_validate_linebyline_valid') 

        while num_positive_samples < num_samples/2:
            sample = next(samples_generator)
            if sample['targets'] == [[1]]:
                samples.append(sample)
                num_positive_samples += 1
            else:
                if len(samples) < num_samples/2:
                    samples.append(sample)
        
    
        num_samples = len(samples)
        y_true = [sample['targets'][0][0] for sample in samples]
        samples = self.process_raw_graphs(samples, is_training_data=False)
        [y_score, y_pred] = self.evaluate_batches(samples)

        #ctr = 0
        #for i in range(num_samples):
        #    if y_true[i] == y_pred[i]:
        #        ctr += 1
        #    print(y_true[i], y_pred[i])
        #print(ctr*1.0/num_samples)
        

        self.print_confusion_matrix(y_true, y_pred)
        print("average_precision:",average_precision_score(y_true, y_pred))
        print("average_precision:",average_precision_score(y_true, y_score[:,1]))

    def read_label_from_file(self, samples, label_file):
        fd = open(label_file,'r')
        labels = fd.read().splitlines()
        y_true = []
        for sample in samples:
            func = sample['func']
            func_id = int(func.strip('.c'))
            offset = func_id - 1
            label = int(labels[offset])
            y_true.append(label)
        return y_true
      

    def evaluate_batches2(self, data):
        fetch_list = [self.node_outputs, tf.nn.softmax(self.node_outputs), tf.argmax(self.output,1)]
        batch_feed_dict = self.make_minibatch_iterator(data, is_training=False)
        
        y_score0 = np.empty(shape=[0,2])
        y_score1 = np.empty(shape=[0,2])
        y_pred = np.empty(shape=[0])

        for item in batch_feed_dict:
            _y_score0 = np.empty(shape=[0,2])
            _y_score1 = np.empty(shape=[0,2])
            _y_pred = np.empty(shape=[0])
            item[self.placeholders['graph_state_keep_prob']] = 1.0
            item[self.placeholders['edge_weight_dropout_keep_prob']] = 1.0
            item[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
            item[self.placeholders['target_values']] = [[]]
            item[self.placeholders['target_mask']] = [[]]
            [_y_score0, _y_score1, _y_pred] = self.sess.run(fetch_list, feed_dict=item)
            y_score0 = np.concatenate((y_score0, _y_score0))
            y_score1 = np.concatenate((y_score1, _y_score1))
            y_pred = np.concatenate((y_pred, _y_pred))

        return [y_score0, y_score1, y_pred] # == [node_scores, node_scores_softmaxed,  y_pred]


    def analyze(self):  
        samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual')
        samples = self.process_raw_graphs(samples, is_training_data=False)
        [node_scores, node_scores_softmaxed, y_pred] = self.evaluate_batches2(samples)

        samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual')
        #samples = self._load_data('/mnt/m1/juliet/ratioN1_80-5-15/cpg_vector_juliet_test')
        ctr = 0
        for idx,s in enumerate(samples):
            num_nodes = len(s['node_features'])
            if s['targets'][0][0] == 1:
                print(y_pred[idx], s['func'])
                s_node_scores = node_scores[ctr:ctr+num_nodes]
                s_node_scores_softmaxed = node_scores_softmaxed[ctr:ctr+num_nodes]
                sorted_ids = s_node_scores[:,1].argsort() #sorted for class1 scores
                top_5_sorted_ids = sorted_ids[-5:]
                for i in top_5_sorted_ids:
                    print(i, s_node_scores[i], s_node_scores_softmaxed[i])

                #s_node_scores = zip(node_scores[ctr:ctr+num_nodes],node_scores_softmaxed[ctr:ctr+num_nodes])
                #for node_idx,scores in enumerate(s_node_scores):
                #    s_node_score = scores[0]
                    #normalized_score_class1 = (s_node_score[1] - s_node_score[0])/math.sqrt(s_node_score[1]*s_node_score[1] + s_node_score[0]*s_node_score[0])
                    #print(node_idx,scores,normalized_score_class1)


                input('press key to continue') 
            ctr += num_nodes

    def example_evaluation2(self):
        ''' Demonstration of what test-time code would look like
        '''
        y_true = []
        y_score = []
        y_pred = []

        #label_file = '/data/sahil/draper/functions_test/labels-CWE-119'
        #label_file = '/data/sahil/draper/functions_test/labels-CWE-119-120'
        label_file = None

        samples = self._load_data('/mnt/m1/devign/cpg_vector_devign_test')
        #samples = self._load_data('/mnt/m1/juliet/cpg_original/ratioN1_80-5-15/cpg_vector_juliet_test')
        #samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual3')
        #samples = self._load_data('/mnt/m1/openssl/cpg_avg_w2v_draper/cpg_vector_openssl')
        #samples = self._load_data('/mnt/m1/juliet/cpg_concat_w2v_draper400/ratioN1_80-5-15/cpg_vector_juliet_test')
        #samples = self._load_data('/data/sahil/cpg_testset/cpg_vector_testset_linebyline')
        #samples = self._load_data('/mnt/m1/cpg_validate_all_plus_train_label119_120/cpg_vector_validate_plus_train_linebyline_valid')
        
        if label_file is not None:
            y_true = self.read_label_from_file(samples,label_file)
        else:    
            for sample in samples:
                y_true.append(sample['targets'][0][0])
        
        samples = self._load_data('/mnt/m1/devign/cpg_vector_devign_test')
        #samples = self._load_data('/mnt/m1/devign/cpg_avg_30_500/cpg_vector_devign_test')
        #samples = self._load_data('/mnt/m1/juliet/cpg_original/ratioN1_80-5-15/cpg_vector_juliet_test')
        #samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual3')
        #samples = self._load_data('/mnt/m1/yufan_dataset/cpg_vector_yufandataset_test')
        #samples = self._load_data('/mnt/m1/openssl/cpg_avg_w2v_draper/cpg_vector_openssl')
        #samples = self._load_data('/mnt/m1/juliet/cpg_concat_w2v_draper400/ratioN1_80-5-15/cpg_vector_juliet_test')
        
        #samples = self._load_data('/data/sahil/cpg_testset/cpg_vector_testset_linebyline')
        #samples = self._load_data('/mnt/m1/cpg_validate_all_plus_train_label119_120/cpg_vector_validate_plus_train_linebyline_valid')
        samples = self.process_raw_graphs(samples, is_training_data=False)

        
        [y_score0, y_score1, y_pred] = self.evaluate_batches(samples)

        #print("accuracy:",accuracy_score(y_true, y_pred))
        #print("confusion matrix:",confusion_matrix(y_true, y_pred))
        #pr, re, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        #print("precision:",pr)
        #print("recall:",re)
        
        self.print_confusion_matrix2(y_true, y_pred)
        print(average_precision_score(y_true, y_score1[:,1]))   #score for postive class
        
        #self.print_confusion_matrix(y_true, y_pred)
        #print("average_precision:",average_precision_score(y_true, y_pred))
        #print("average_precision:",average_precision_score(y_true, y_score0[:,1]))   #score for postive class
        #print("average_precision_scaled:",average_precision_score(y_true, y_score1[:,1]))   #score for postive class

        #samples = self._load_data('/data/sahil/validate_plus_train_label1/cpg_vector_validate_plus_train_label1_linebyline_valid')
        #self.print_miscls(y_true, y_pred, samples)

    
    def read_labels_from_files(self, samples, label_files):
        multi_labels = []
        for label_file in label_files:
            fd = open(label_file,'r')
            labels = list(map(lambda x: int(x),fd.read().splitlines()))
            multi_labels.append(labels)
        labels = list(zip(*multi_labels))

        y_true = []
        for sample in samples:
            func = sample['func']
            func_id = int(func.strip('.c'))
            offset = func_id - 1
            label = labels[offset]
            y_true.append(label)
        return y_true   #[(sample_0_label_0, sample_0_label_1), ..... , (sample_N_label_0, sample_N_label_1)] assuming 2 class
    
    def print_miscls2(self):
        samples = self._load_data('/data/sahil/cpg_testset/cpg_vector_testset_linebyline')
        label_file119 = '/data/sahil/draper/functions_test/labels-CWE-119'
        label_file120 = '/data/sahil/draper/functions_test/labels-CWE-120'
        label_file469 = '/data/sahil/draper/functions_test/labels-CWE-469'
        label_file476 = '/data/sahil/draper/functions_test/labels-CWE-476'
        label_fileXXX = '/data/sahil/draper/functions_test/labels-CWE-other'
        label_files = [label_file119, label_file120, label_file469, label_file476, label_fileXXX]
        Y_true = self.read_labels_from_files(samples, label_files)
        
        samples = self._load_data('/data/sahil/cpg_testset/cpg_vector_testset_linebyline')
        samples = self.process_raw_graphs(samples, is_training_data=False)
        [Y_score0, Y_score1, Y_pred_119] = self.evaluate_batches(samples)

        Y_true_119 = [i[0] for i in Y_true]
        self.print_confusion_matrix(Y_true_119, Y_pred_119)
        
        ctr_119FN_othersTP = [0 for i in label_files]
        ctr_119FP_othersTN = [0 for i in label_files]
        for idx,y_pred_119 in enumerate(Y_pred_119):
            y_true = Y_true[idx]
            y_true_119 = y_true[0]
            if y_pred_119 != y_true_119:
                if y_pred_119 == 0:
                    for idx,y_t in enumerate(y_true):
                        if y_t == 1:
                            ctr_119FN_othersTP[idx] += 1
                else:
                    for idx,y_t in enumerate(y_true):
                        if y_t == 0:
                            ctr_119FP_othersTN[idx] += 1
                    
        print("ctr_119FN_othersTP:",ctr_119FN_othersTP)
        print("ctr_119FP_othersTN:",ctr_119FP_othersTN)


    def print_miscls1(self):
        samples = self._load_data('/data/sahil/cpg_testset/cpg_vector_testset_linebyline')
        label_file119 = '/data/sahil/draper/functions_test/labels-CWE-119'
        label_file120 = '/data/sahil/draper/functions_test/labels-CWE-120'
        Y_true = self.read_labels_from_files(samples,[label_file119, label_file120])
        
        samples = self._load_data('/data/sahil/cpg_testset/cpg_vector_testset_linebyline')
        samples = self.process_raw_graphs(samples, is_training_data=False)
        [Y_score0, Y_score1, Y_pred_119] = self.evaluate_batches(samples)

        Y_true_119 = [i for (i,j) in Y_true]
        self.print_confusion_matrix(Y_true_119, Y_pred_119)
        
        ctr_119FN_120TP = 0
        ctr_119FP_120TN = 0
        ctr_F = 0
        for idx,y_pred_119 in enumerate(Y_pred_119):
            y_true_119, y_true_120 = Y_true[idx]
            if y_pred_119 != y_true_119:
                if y_pred_119 == 0 and y_true_120 == 1:
                   ctr_119FN_120TP += 1
                elif y_pred_119 == 1 and y_true_120 == 0:
                    ctr_119FP_120TN += 1
                else:
                    ctr_F += 1
                    
        print("ctr_119FN_120TP:",ctr_119FN_120TP)
        print("ctr_119FP_120TN:",ctr_119FP_120TN)
        print("ctr_F:",ctr_F)
        


    def print_miscls(self, y_true, y_pred, samples):
        ctr = 0
        for idx,s in enumerate(samples):
            if s['targets'][0][0] == 1:
                if y_pred[idx] == y_true[idx]:
                    print(s['func'])
                    ctr += 1
        print(ctr)        


    def print_miscls3(self):
        #samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual')
        #samples = self._load_data('/mnt/m1/juliet/cpg_concat/ratioN1_80-5-15/cpg_vector_juliet_test')
        #samples = self._load_data('/mnt/m1/openssl/cpg_avg_w2v_draper/cpg_vector_openssl')
        samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual3')
        samples = self.process_raw_graphs(samples, is_training_data=False)
        [y_scores, y_scores_softmaxed, y_pred] = self.evaluate_batches(samples)

        #samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual')
        #samples = self._load_data('/mnt/m1/juliet/cpg_concat/ratioN1_80-5-15/cpg_vector_juliet_test')
        #samples = self._load_data('/mnt/m1/openssl/cpg_avg_w2v_draper/cpg_vector_openssl')
        samples = self._load_data('/mnt/m1/sbabi/cpg_vector_test_manual3')
        ctr = 0
        for idx,s in enumerate(samples):
            #if s['targets'][0][0] == 1 and y_pred[idx] != 1:
                print(y_pred[idx], y_scores[idx], y_scores_softmaxed[idx], s['src'] + "/" + s['func'])
                ctr += 1
        print(ctr)

def main():
    args = docopt(__doc__)
    try:
        model = SparseGGNNChemModel(args)
        if args['--evaluate']:
            #model.analyze()
            model.example_evaluation2()
            #model.print_miscls2()
            #model.print_miscls3()
        else:
            model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()
