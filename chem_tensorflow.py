#!/usr/bin/env/python

from typing import Tuple, List, Any, Sequence

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random
import math
from sklearn.metrics import average_precision_score


from utils import MLP, ThreadedIterator, SMALL_NUMBER


class ChemModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 3000,
            'patience': 50,
            'learning_rate': 0.0001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 0.8,

            'hidden_size': 300,
            'num_timesteps': 8,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,

            #'train_file': '/mnt/m1/juliet/cpg_concat_w2v_draper400/ratioN1_80-5-15/cpg_vector_juliet_train',
            #'valid_file': '/mnt/m1/juliet/cpg_concat_w2v_draper400/ratioN1_80-5-15/cpg_vector_juliet_valid'
            #'train_file': '/home/suneja/data/cpg_validate_plus_train_label119_ratio11/cpg_vector_validate_plus_train_linebyline_train',
            #'valid_file': '/home/suneja/data/cpg_validate_plus_train_label119_ratio11/cpg_vector_validate_plus_train_linebyline_valid'
            #'train_file': '/mnt/m1/cpg_validate_all_plus_train_label119_120/cpg_vector_validate_plus_train_linebyline_train',
            #'valid_file': '/mnt/m1/cpg_validate_all_plus_train_label119_120/cpg_vector_validate_plus_train_linebyline_valid',
            #'train_file': '/mnt/m1/sbabi/cpg_vector_train',
            #'valid_file': '/mnt/m1/sbabi/cpg_vector_valid'
            #'train_file': '/mnt/m1/juliet/cpg_original/ratioN1_80-5-15/cpg_vector_juliet_train',
            #'valid_file': '/mnt/m1/juliet/cpg_original/ratioN1_80-5-15/cpg_vector_juliet_valid'
            'train_file': '/mnt/m1/devign/cpg_vector_devign_train',
            'valid_file': '/mnt/m1/devign/cpg_vector_devign_valid'
        }

    def __init__(self, args):
        self.args = args

        # Collect argument things:
        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)
        self.best_model_file1 = os.path.join(log_dir, "%s_model_best_loss.pickle" % self.run_id)
        self.best_model_file2 = os.path.join(log_dir, "%s_model_best_ap.pickle" % self.run_id)

        # Collect parameters:
        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])


        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.data_file_offsets_class0 = []
        self.data_file_offsets_class1 = []
        self.class1_weight = 1
        self.class0_weight = 1
        self.load_data_stats_cached(params['train_file'])
        if not self.data_file_offsets_class0 or not self.data_file_offsets_class1 or not self.max_num_vertices or not self.num_edge_types or not self.annotation_size:
            self.load_data_stats(params['train_file'], is_training_data=True)
            self.load_data_stats(params['valid_file'], is_training_data=False)
            self.save_data_stats(params['train_file'])
        self._valid_data = []
        self._train_data = {}
        #self._train_data = [{} for i in range(len(self.data_file_offsets_class0) + len(self.data_file_offsets_class1))]

        # Build the actual model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(params['random_seed'])
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            # Restore/initialize variables:
            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

    def _load_data_stats(self, filename):
        offset = 0
        offsets_class0 = []
        offsets_class1 = []
        num_samples_class0 = 0
        num_samples_class1 = 0
        
        fd = open(filename,'r')
        for line in fd:
            sample = json.loads(line)
            sample_label = sample["targets"][0][0]  #[[0]] or [[1]]
            if sample_label == 1:
                offsets_class1.append(offset)   #separating offsets to maintain class ratio in batches
                num_samples_class1 += 1
            else:
                offsets_class0.append(offset)
                num_samples_class0 += 1
            offset += len(line)
            yield sample
        fd.close()

        self.data_file_offsets_class0 = offsets_class0
        self.data_file_offsets_class1 = offsets_class1
        self.num_samples_class0 = num_samples_class0
        self.num_samples_class1 = num_samples_class1

        #to save time for future iters
        fd = open(filename+'_offsets_class0','w')
        for offset in offsets_class0:
            fd.write(str(offset) + '\n')
        fd.close()    

        fd = open(filename+'_offsets_class1','w')
        for offset in offsets_class1:
            fd.write(str(offset) + '\n')
        fd.close()    


    def _load_data(self, filename):
        fd = open(filename,'r')
        for line in fd:
            yield json.loads(line)
        fd.close()
    
    def _load_valid_data_mem(self, filename):
        if self._valid_data == []:
            fd = open(filename,'r')
            self._valid_data = [json.loads(line) for line in fd]
            fd.close()
        for i in self._valid_data:
            yield i
   

    def shuffle_classes(self):
        #assumption less number os positive samples
        num_samples_class0 = self.num_samples_class0
        num_samples_class1 = self.num_samples_class1
        ratio = math.ceil(num_samples_class0/num_samples_class1) 
            #this makes last batch full of ones (about last 300 items in classids are 1),
            #can alternately use math.floor with more ones per batch ... 
            #... then drop the last few batches (turned out to be about 2400 zeroes)
        ctr0 = 0
        ctr1 = 0
        end = ratio
        classids = []

        while ctr0 < num_samples_class0 or ctr1 < num_samples_class1:
            _classids = []
            if ctr0 < num_samples_class0:
                if ctr0 + end > num_samples_class0:
                    end = num_samples_class0 - ctr0
                _classids = [0 for item in range(0, end)]
                ctr0 += end
            if ctr1 < num_samples_class1:
                _classids.append(1)
                ctr1 += 1
            random.shuffle(_classids)
            classids += _classids
        return classids


    def _load_data_shuffled(self, filename):  #reading in random order as opposed to in-mem shuffling for efficiency
        classids = self.shuffle_classes()    
        random.shuffle(self.data_file_offsets_class0)
        random.shuffle(self.data_file_offsets_class1)
        ctr0 = 0
        ctr1 = 0

        fd = open(filename,'r')
        for classid in classids:
            if classid == 0:
                offset = self.data_file_offsets_class0[ctr0]
                ctr0 += 1
            else:
                offset = self.data_file_offsets_class1[ctr1]
                ctr1 += 1
            fd.seek(offset)
            line = fd.readline()
            yield json.loads(line)
        fd.close()


    def _load_train_data_mem(self,filename):
        classids = self.shuffle_classes()    
        random.shuffle(self.data_file_offsets_class0)
        random.shuffle(self.data_file_offsets_class1)
        ctr0 = 0
        ctr1 = 0

        fd = open(filename,'r')
        for classid in classids:
            if classid == 0:
                offset = self.data_file_offsets_class0[ctr0]
                ctr0 += 1
            else:
                offset = self.data_file_offsets_class1[ctr1]
                ctr1 += 1
            if  offset not in self._train_data:   #easier done while loading stats, but this was done after the fact 
                fd.seek(offset)
                line = fd.readline()
                self._train_data[offset]  = json.loads(line)
            yield self._train_data[offset]    
        fd.close()


    def get_data(self, file_name, is_training_data: bool):  #multi generators for mem efficiency
        #full_path = os.path.join(self.data_dir, file_name)
        full_path = file_name
        if is_training_data:
            #data = self._load_data_shuffled(full_path) 
            data = self._load_train_data_mem(full_path) 
        else:
            #data = self._load_data(full_path) 
            data = self._load_valid_data_mem(full_path) 
        return self.process_raw_graphs(data, is_training_data)


    def load_data_stats_cached(self, filename):
        if not os.path.exists(filename+'_offsets_class0') or not os.path.exists(filename+'_offsets_class1'):
            return

        fd = open(filename+'_offsets_class0','r')
        self.data_file_offsets_class0 = [int(offset) for offset in fd.readlines()]
        assert len(self.data_file_offsets_class0) > 100
        fd.close()
        
        fd = open(filename+'_offsets_class1','r')
        self.data_file_offsets_class1 = [int(offset) for offset in fd.readlines()]
        assert len(self.data_file_offsets_class1) > 100
        fd.close()

        fd = open(filename+'_stats','r')
        stats = [int(item) for item in fd.readlines()]
        assert len(stats) == 5
        self.max_num_vertices = stats[0]
        self.num_edge_types = stats[1]
        self.annotation_size = stats[2]
        self.num_samples_class0 = stats[3]
        self.num_samples_class1 = stats[4]

        #ratio = 1.0 * self.num_samples_class1 / (self.num_samples_class0 + self.num_samples_class1)
        #self.class1_weight = 1-ratio
        #self.class0_weight = ratio
        
        self.class1_weight = 1
        self.class0_weight = 1

        fd.close()


    def save_data_stats(self, filename):
        fd = open(filename+'_stats','w')
        fd.write(str(self.max_num_vertices) + '\n')
        fd.write(str(self.num_edge_types) + '\n')
        fd.write(str(self.annotation_size) + '\n')
        fd.write(str(self.num_samples_class0) + '\n')
        fd.write(str(self.num_samples_class1) + '\n')
        fd.close()


    def load_data_stats(self, file_name, is_training_data: bool):
        #full_path = os.path.join(self.data_dir, file_name)
        full_path = file_name
        print("Loading data from %s" % full_path)
        if is_training_data:
            data_1 = self._load_data_stats(full_path) #multi generators for mem efficiency
        else:
            data_1 = self._load_data(full_path) #multi generators for mem efficiency
        data_2 = self._load_data(full_path)

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data_1:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        data_item = next(data_2)
        self.annotation_size = max(self.annotation_size, len(data_item["node_features"][0]))
        #self.annotation_size = max(self.annotation_size, len(data_2[0]["node_features"][0]))
        return


    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        #self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
        #                                                    name='target_values')
        self.placeholders['target_values'] = tf.placeholder(tf.int64, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['loss_weights'] = tf.placeholder(tf.float32, [None], name='loss_weights_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

#        self.ops['losses'] = []
#        for (internal_id, task_id) in enumerate(self.params['task_ids']):
#            with tf.variable_scope("out_layer_task%i" % task_id):
#                with tf.variable_scope("regression_gate"):
#                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
#                                                                           self.placeholders['out_layer_dropout_keep_prob'])
#                with tf.variable_scope("regression"):
#                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
#                                                                                self.placeholders['out_layer_dropout_keep_prob'])
#                computed_values = self.gated_regression(self.ops['final_node_representations'],
#                                                        self.weights['regression_gate_task%i' % task_id],
#                                                        self.weights['regression_transform_task%i' % task_id])
#                diff = computed_values - self.placeholders['target_values'][internal_id,:]
#                task_target_mask = self.placeholders['target_mask'][internal_id,:]
#                task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
#                diff = diff * task_target_mask  # Mask out unused values
#                self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
#                task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
#                # Normalise loss to account for fewer task-specific examples in batch:
#                task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
#                self.ops['losses'].append(task_loss)
#        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])
       
        task_id = 0
        internal_id = 0
        with tf.variable_scope("out_layer_task%i" % task_id):
            target_values = self.placeholders['target_values'][internal_id,:]
            loss_weights = self.placeholders['loss_weights']
            with tf.variable_scope("classification_gate"):
                self.weights['classification_gate'] = MLP(2 * self.params['hidden_size'], 2, [], self.placeholders['out_layer_dropout_keep_prob'])
            with tf.variable_scope("classification"):
                self.weights['classification_transform'] = MLP(self.params['hidden_size'], 2, [],self.placeholders['out_layer_dropout_keep_prob'])
            computed_values = self.classification_task(self.ops['final_node_representations'], self.weights['classification_gate'], self.weights['classification_transform'])
            #computed_values = self.classification_task(self.ops['final_node_representations'], self.weights['classification_task'])
            #loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=target_values, logits=computed_values, weights=loss_weights, reduction=tf.losses.Reduction.NONE)) 
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=computed_values,labels=target_values)) 

            #loss_weights = tf.ones(self.placeholders['num_graphs'])
            #target_values_one_hot = tf.one_hot(target_values,2)
            #loss = self.focal_loss(computed_values, target_values_one_hot, loss_weights)
            #loss = loss/tf.cast(self.placeholders['num_graphs'],tf.float32)

            #TODO maybe:sigmoid_cross_entropy_with_logits
            prediction = tf.argmax(computed_values,1)  
            accuracy = tf.equal(prediction, target_values)
            accuracy = tf.reduce_mean(tf.cast(accuracy,"float"))

            #pred_prob = tf.nn.softmax(computed_values)
            #self.ops['target_values'] = target_values
            #self.ops['pred_prob'] = pred_prob[:,1]

            self.ops['accuracy_task%i' % task_id] = accuracy
            self.ops['loss_task'] = loss 
            #self.ops['optimizer'] = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)
            #self.ops['computed_values'] = computed_values
            #self.ops['target_values'] = target_values
            #self.ops['prediction'] = prediction
            
            #computed_values = self.classification_task(self.ops['final_node_representations'])
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=computed_values,labels=target_values)) 
            #target_values_int = tf.cast(target_values, dtype=tf.int32)
            #prediction = tf.round(tf.sigmoid(computed_values))  #TODO: try softmax
                
            #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=computed_values,labels=target_values)) 
        self.ops['loss'] = self.ops['loss_task']

    def focal_loss(self, logits, onehot_labels, weights, alpha=0.25, gamma=2.0):
        with tf.name_scope("focal_loss"):
            logits = tf.cast(logits, tf.float32)
            onehot_labels = tf.cast(onehot_labels, tf.float32)
            ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
            predictions = tf.sigmoid(logits)
            predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
            # add small value to avoid 0
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
            weighted_loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t * tf.expand_dims(weights, axis=1)
            return tf.reduce_sum(weighted_loss)


    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")
    
    def classification_task(self, last_h, classification_layer):
    #def classification_task(self, last_h):
        raise Exception("Models have to implement classification_task!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        average_precision = 0
        accuracies = []
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        start_time = time.time()
        processed_graphs = 0
        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                #fetch_list = [self.ops['loss'], accuracy_ops, self.ops['target_values'], self.ops['pred_prob'], self.ops['optimizer']]
                #fetch_list = [self.ops['loss'], accuracy_ops, self.ops['optimizer']]
                fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step']]
                #fetch_list = [self.ops['loss'], accuracy_ops, self.ops['optimizer'], self.ops['computed_values'], self.ops['target_values'], self.ops['prediction']]
                #fetch_list = [self.ops['loss'], accuracy_ops, self.ops['train_step'], self.ops['computed_values'], self.ops['target_values'], self.ops['prediction']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                #fetch_list = [self.ops['loss'], accuracy_ops, self.ops['target_values'], self.ops['pred_prob']]
                fetch_list = [self.ops['loss'], accuracy_ops]
            result = self.sess.run(fetch_list, feed_dict=batch_data)
            (batch_loss, batch_accuracies) = (result[0], result[1])
            #(batch_loss, batch_accuracies, batch_target_values, batch_pred_prob) = (result[0], result[1], result[2], result[3])
            loss += batch_loss * num_graphs
            #ap = average_precision_score(batch_target_values, batch_pred_prob)
            #average_precision += ap * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)

            print("Running %s, batch %i (has %i graphs). Loss so far: %.4f" % (epoch_name,
                                                                               step,
                                                                               num_graphs,
                                                                               loss / processed_graphs),
                  end='\r')

            #x = self.sess.run(self.output)      
            #prediction = result[-1]
            #target_values = result[-2]
            #computed_values = result[-3]
            #import pdb
            #pdb.set_trace()
    
        print("Num graphs proessed:", processed_graphs)
        #average_precision = average_precision / processed_graphs 
        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)
        return loss, average_precision, accuracies, error_ratios, instance_per_sec

    def train(self):
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                self.valid_data = self.get_data(self.params['valid_file'], is_training_data=False)
                valid_loss, valid_ap, valid_accs, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, False)
                best_val_acc = np.sum(valid_accs)
                best_val_loss = np.sum(valid_loss)
                best_val_ap = valid_ap
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
                print("\r\x1b[KResumed operation, initial cum. val. loss: %.5f" % best_val_loss)
            else:
                #(best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
                (best_val_acc, best_val_acc_epoch) = (0.0, 0)
                best_val_loss = 1000
                best_val_ap = 0
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                self.train_data = self.get_data(self.params['train_file'], is_training_data=True)
                self.valid_data = self.get_data(self.params['valid_file'], is_training_data=False)
                train_loss, train_ap, train_accs, train_errs, train_speed = self.run_epoch("epoch %i (training)" % epoch,
                                                                                 self.train_data, True)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], train_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], train_errs)])
                print("\r\x1b[K Train: loss: %.5f | ap: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (train_loss,
                                                                                                        train_ap,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        train_speed))
                valid_loss, val_ap, valid_accs, valid_errs, valid_speed = self.run_epoch("epoch %i (validation)" % epoch,
                                                                                 self.valid_data, False)
                accs_str = " ".join(["%i:%.5f" % (id, acc) for (id, acc) in zip(self.params['task_ids'], valid_accs)])
                errs_str = " ".join(["%i:%.5f" % (id, err) for (id, err) in zip(self.params['task_ids'], valid_errs)])
                print("\r\x1b[K Valid: loss: %.5f | ap: %.5f | acc: %s | error_ratio: %s | instances/sec: %.2f" % (valid_loss,
                                                                                                        val_ap,
                                                                                                        accs_str,
                                                                                                        errs_str,
                                                                                                        valid_speed))

                epoch_time = time.time() - total_time_start
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time,
                    'train_results': (train_loss, train_accs.tolist(), train_errs.tolist(), train_speed),
                    'valid_results': (valid_loss, valid_accs.tolist(), valid_errs.tolist(), valid_speed),
                }
                log_to_save.append(log_entry)
                with open(self.log_file, 'w') as f:
                    json.dump(log_to_save, f, indent=4)
                
                if val_ap > best_val_ap:
                    self.save_model(self.best_model_file2)
                    print("  (Best average_precision epoch so far, cum. val. ap increased to %.5f from %.5f. Saving to '%s')" % (val_ap, best_val_ap, self.best_model_file2))
                    best_val_ap = val_ap
            
                val_loss = np.sum(valid_loss)  # type: float
                if val_loss < best_val_loss: 
                    self.save_model(self.best_model_file1)
                    print("  (Best loss epoch so far, cum. val. loss decreased to %.5f from %.5f. Saving to '%s')" % (val_loss, best_val_loss, self.best_model_file1))
                    best_val_loss = val_loss

                val_acc = np.sum(valid_accs)  # type: float
                
                #if val_acc < best_val_acc: # For the org code where accuracy meant error; lower is better
                if val_acc > best_val_acc: # For new code where accuracy means accuracy, higher is better
                    self.save_model(self.best_model_file)
                    #print("  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to '%s')" % (val_acc, best_val_acc, self.best_model_file))
                    print("  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')" % (val_acc, best_val_acc, self.best_model_file))
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                elif epoch - best_val_acc_epoch >= self.params['patience']:
                    print("Stopping training after %i epochs without improvement on validation accuracy." % self.params['patience'])
                    break

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
                         "params": self.params,
                         "weights": weights_to_save
                       }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        #SAHIL: skipping this coz assertion error when data paths dont match after reorg
        # Assert that we got the same model configuration
        #assert len(self.params) == len(data_to_load['params'])
        #for (par, par_value) in self.params.items():
            # Fine to have different task_ids:
        #    if par not in ['task_ids', 'num_epochs']:
        #        assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)
