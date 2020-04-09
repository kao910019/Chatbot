# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import codecs
import logging
import warnings

import numpy as np
# Disable numpy warning.
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore',category=FutureWarning)
# Disable tensorflow warning.
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_COND_V2'] = '1'
# import nltk
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import timeline

from tqdm import tqdm

from hparams import HParams
from model import Transformer
from data_process import DataLoader
from modules import variable_loader, time_string
#Load settings file
from settings import SYSTEM_ROOT
from settings import RESULT_DIR, RESULT_FILE, TRAIN_LOG_DIR, TEST_LOG_DIR, INFER_LOG_DIR, METADATA_FILE, VOCAB_FILE
from settings import TRAIN_MODE_LIST
#from settings import EMOTION_TYPES, EMOTION_LENGTH
            
class Interface(object):
    def __init__(self, training, hparams = None):
        self.training = training
        print("# Prepare dataset placeholder and hyper parameters ...")
        #Load Hyper-parameters file
        self.hparams = hparams if hparams else HParams().hparams
        
        if self.hparams.use_gpu:
            self.config = tf.ConfigProto(allow_soft_placement=True)
            tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            self.config.gpu_options.allow_growth = True
        else:
            self.config = None
            
        
        # Enable cluster server
        self.enable_cluster = (self.hparams.job_name != None)
        self.is_chief = (self.hparams.task_index == 0)
        if self.enable_cluster:
            self.build_server()
        else:
            with tf.Graph().as_default() as graph:
                self.session1 = tf.Session(config=self.config)
#                self.session2 = tf.Session(config=self.config)
                self.graph = graph
                self.build_env()
                self.train()
            
# =============================================================================
# Create model
# =============================================================================
# python network_interface.py --job_name=ps --task_index=0 > ps0.log &
# python network_interface.py --job_name=worker --task_index=0 > worker0.log
# python network_interface.py --job_name=worker --task_index=1 > worker1.log
    def build_server(self):
        self.params_server = self.hparams.ps_hosts.split(",")
        self.worker_server = self.hparams.wk_hosts.split(",")
        self.cluster = tf.train.ClusterSpec({"ps": self.params_server, "worker": self.worker_server})
        self.server = tf.train.Server(self.cluster, job_name=self.hparams.job_name, task_index=self.hparams.task_index)
        if self.hparams.job_name == 'ps':
            print("# Params Server Started.")
            self.server.join()
        self.worker_device = '/job:worker/task:{}/cpu:0'.format(self.hparams.task_index)
        with tf.device(tf.train.replica_device_setter(cluster=self.cluster, worker_device=self.worker_device)):
            with tf.Graph().as_default() as graph:
                self.graph = graph
                self.build_env()
                
                self.hooks1 = [tf.train.CheckpointSaverHook(checkpoint_dir=os.path.join(RESULT_DIR, 'session1'), save_steps=250)]
#                               tf.train.SummarySaverHook(save_steps=1, summary_writer=self.train_summary_writer, summary_op=self.model.step_summary)]
#                self.hooks2 = [tf.train.CheckpointSaverHook(checkpoint_dir=os.path.join(RESULT_DIR, 'session2'), save_secs=18000)]
                
#                for i in tf.contrib.framework.list_variables(os.path.join(RESULT_DIR, 'session2')):
#                    print(i)
#                    
#                sys.exit()
                
                if self.hparams.num_steps != 0:
                    self.hooks.append(tf.train.StopAtStepHook(last_step=self.hparams.num_steps))
                    
                if self.is_chief:
                    print( '# Worker {}: Initailizing session...'.format(self.hparams.task_index))
                else:
                    print( '# Worker {}: Waiting for session to be initaialized...'.format(self.hparams.task_index))
                
                self.session1 = tf.train.MonitoredTrainingSession(
                        master=self.server.target, is_chief=self.is_chief, checkpoint_dir=os.path.join(RESULT_DIR, 'session1'), config=self.config, hooks=self.hooks1, save_summaries_steps=None, save_summaries_secs=None)
                
#                self.session2 = tf.train.MonitoredTrainingSession(
#                        master=self.server.target, is_chief=self.is_chief, checkpoint_dir=os.path.join(RESULT_DIR, 'session2'), config=self.config, hooks=self.hooks2, save_summaries_steps=None, save_summaries_secs=None)
                
                self.train()
                
#                with tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=self.is_chief, 
#                                                       checkpoint_dir=RESULT_DIR, hooks=self.hooks,
#                                                       save_summaries_steps=None, save_summaries_secs=None) as session:
#                    self.session = session
#                    self.train()
                
    def build_env(self):
        if self.training:
            self.build_train_model()
            self.run_options, self.run_metadata = None, None
            if self.hparams.trace_timeline:
            # Use 'chrome://tracing/' to load timeline_client.json file
                self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                self.run_metadata = tf.RunMetadata()
            #tensorboard
            self.train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR+self.train_mode+str(self.hparams.task_index), self.graph)
            self.test_summary_writer = tf.summary.FileWriter(TEST_LOG_DIR+self.train_mode+str(self.hparams.task_index)) if self.hparams.test_dataset else None
            #tensorboard projector
            config = projector.ProjectorConfig()
            embed = config.embeddings.add()
            embed.tensor_name = "Network/vocab_embeddings/embedding"
            if not os.path.isfile(METADATA_FILE):
                with codecs.open(VOCAB_FILE,encoding='utf-8', mode='r') as f:
                    data = f.read()
                with codecs.open(METADATA_FILE, encoding='utf-8', mode='w') as f:
                    f.write(data)
            embed.metadata_path = METADATA_FILE
            projector.visualize_embeddings(self.train_summary_writer, config)
        else:
            self.build_predict_model()
            #Tensorboard
            tf.summary.FileWriter(INFER_LOG_DIR, self.graph)
            
    def build_train_model(self):
        self.train_mode = None
        print("# Select train mode [{}]".format("/".join([i[:3] for i in TRAIN_MODE_LIST])))
        for mode in TRAIN_MODE_LIST:
            if mode.startswith(self.hparams.train_mode):
                self.train_mode = mode
        assert self.train_mode
        
        self.data_loader = DataLoader(hparams = self.hparams, training = self.training, mode = self.train_mode)
        
        with tf.variable_scope('Network_Operator'):
            self.dataset_handler = tf.placeholder(tf.string, shape=[], name='dataset_handler')
            self.train_batch_iter = self.data_loader.get_training_batch(self.data_loader.train_dataset)
            self.test_batch_iter = self.data_loader.get_training_batch(self.data_loader.test_dataset)
            self.train_dataset_count, self.test_dataset_count = self.data_loader.train_dataset_count, self.data_loader.test_dataset_count
            input_batch = self.data_loader.multiple_batch(self.dataset_handler, self.train_batch_iter.batched_dataset)
        
        print("# Build model =", self.train_mode)
        self.model = Transformer(mode = self.train_mode,
                                 graph = self.graph,
                                 hparams = self.hparams,
                                 data_loader = self.data_loader,
                                 batch_input = input_batch)
        
        self.global_step = self.model.global_step
        self.epoch_num = self.model.train_epoch
# =============================================================================
# Run model
# =============================================================================
    def train(self):
        #-----------------------------------------------------------
        print("# Get dataset handler.")
        # Training handler
        training_handle = self.session1.run(self.train_batch_iter.handle)
        dataset_dict = [{'tag':'Train', 'dataset_count': self.train_dataset_count, 'writer':self.train_summary_writer, 
                         'handler':{self.dataset_handler: training_handle}, 'initer':self.train_batch_iter.initializer}]
        init_list = [self.train_batch_iter.initializer]
        # Testing handler
        if self.hparams.test_dataset and self.is_chief:
            testing_handle = self.session1.run(self.test_batch_iter.handle)
            dataset_dict.append({'tag':'Test', 'dataset_count': self.test_dataset_count, 'writer':self.test_summary_writer, 
                                 'handler':{self.dataset_handler: testing_handle}, 'initer':self.test_batch_iter.initializer})
            init_list.append(self.test_batch_iter.initializer)
            
#        self.model.build_special_summary()
        #-----------------------------------------------------------
        print("# Initialize all Variable.")
        self.session1.run(init_list)
#        self.session2.run(init_list)
        
        if not self.enable_cluster:
            self.session1.run(tf.tables_initializer())
            self.session1.run(tf.global_variables_initializer(), feed_dict = dataset_dict[0]['handler'])
            var_list = [var for var in tf.global_variables() if var.name not in self.model.unsave_variable_namelist]
            self.saver1, _ = variable_loader(self.session1, os.path.join(RESULT_DIR, 'session1'), var_list = var_list, max_to_keep = self.hparams.max_to_keep)
            
            
#            self.session2.run(tf.tables_initializer())
#            self.session2.run(tf.global_variables_initializer(), feed_dict = dataset_dict[0]['handler'])
#            var_list = [var for var in tf.global_variables() if var.name not in self.model.unsave_variable_namelist]
#            self.saver2, _ = variable_loader(self.session2, os.path.join(RESULT_DIR, 'session2'), var_list = var_list, max_to_keep = self.hparams.max_to_keep)
        
        
        
        train_epoch_times = 0
        global_step = self.global_step.eval(session=self.session1)
        epoch_num = self.epoch_num.eval(session=self.session1)
        
        if self.train_mode == 'ddpg' and self.hparams.force_replace:
            print("# Initialize target params from eval params.")
            self.session1.run(self.model.force_replace)
        
        print("="*30)
        print("# Global step = {}".format(global_step))
        print("# Training loop started @ {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        print("# Epoch training {} times.".format(self.hparams.num_epochs))
        
        server_stop_flag = False if not self.enable_cluster else self.session1.should_stop()
        while (train_epoch_times < self.hparams.num_epochs) and not server_stop_flag:
            
            for index in range(len(dataset_dict)):
                print("# Start {} step.".format(dataset_dict[index]['tag']))
                      
                iter_times = math.ceil(dataset_dict[index]['dataset_count'] / self.hparams.batch_size)
                process_bar = tqdm(total = iter_times, leave=False)
                epoch_start_time = time.time()
                self.session1.run(init_list)
#                self.session2.run(init_list)
                    
                while not server_stop_flag:
                    try:
                        # Evluation during training
                        evaluation = True if (global_step % 250 == 0 or process_bar.n == int(iter_times/2)) else False
                        update = True if dataset_dict[index]['tag'] == 'Train' else False
                        
                        global_step = self.model.update(self.session1, self.session1, dataset_dict[index]['handler'], writer = dataset_dict[index]['writer'], 
                                                        update = update, evaluation = evaluation, options=self.run_options, run_metadata=self.run_metadata)
                        # Predict LastTime and process bar
                        process_bar.update(1)
                        summary = self.session1.run(self.model.one_batch_per_sec_summary, feed_dict={self.model.one_batch_per_sec: process_bar.format_dict['elapsed']/process_bar.format_dict['n']})
                        dataset_dict[index]['writer'].add_summary(summary, global_step)
                        
#                        if (global_step % 250 == 0) and not self.enable_cluster:
#                            self.saver1.save(self.session1, os.path.join(os.path.join(RESULT_DIR, 'session1'), "model"), global_step = global_step)
#                            self.saver2.save(self.session2, os.path.join(os.path.join(RESULT_DIR, 'session2'), "model"), global_step = global_step)
                            
                    # End of each epoch
                    except tf.errors.OutOfRangeError:
                        # Total time
                        epoch_dur = time_string(time.time() - epoch_start_time)
                        
                        if self.is_chief:
                            self.model.write_epoch_summary(self.session1, dataset_dict[index]['writer'], epoch_num)
                        train_epoch_times += 1
                        epoch_num = self.session1.run(self.model.assign_train_epoch)
                        if dataset_dict[index]['tag'] == 'Test' and not self.enable_cluster:
                            self.saver1.save(self.session1, os.path.join(os.path.join(RESULT_DIR, 'session1'), "model"), global_step = global_step)
#                            self.saver2.save(self.session2, os.path.join(os.path.join(RESULT_DIR, 'session2'), "model"), global_step = global_step)
                            
                        if self.hparams.trace_timeline:
                            # Create the Timeline object, and write it to a json file
                            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open('timeline_{}.json'.format(self.hparams.task_index), 'w') as f:
                                f.write(chrome_trace)
                            
                        print("# {} step {:5d} @ {} | {} elapsed."
                          .format(dataset_dict[index]['tag'], global_step, time.strftime("%Y-%m-%d %H:%M:%S"), epoch_dur))
                        break
                process_bar.close()
            print("# Finished epoch {:2d}/{:2d} @ {} | {} elapsed."
              .format(train_epoch_times, self.hparams.num_epochs, time.strftime("%Y-%m-%d %H:%M:%S"), epoch_dur))
            
        # END the Training.
        self.train_summary_writer.close()
        if self.hparams.test_dataset:
            self.test_summary_writer.close()
    
    def build_predict_model(self):
        self.src_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name = 'Inputs')
        self.src_length_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name = 'Inputs_length')
        src_dataset = tf.data.Dataset.from_tensor_slices((self.src_placeholder, self.src_length_placeholder))
        self.infer_batch = self.data_loader.get_inference_batch(src_dataset)
        print("# Build inference model ...")
        self.model = Transformer(mode = 'inference',
                                 graph = self.graph,
                                 hparams = self.hparams,
                                 data_loader = self.data_loader,
                                 batch_input = self.infer_batch)
        print("# Restoring model weights ...")
        self.saver, self.restore = variable_loader(self.session, RESULT_DIR)
        assert self.restore
        self.session.run(tf.tables_initializer())
        
    # def generate(self, question):
    #     tokens = nltk.word_tokenize(question.lower())
    #     sentence = [' '.join(tokens[:]).strip()]
    #     length = len(sentence[0].split()) if question != '_non_' else 0
        
    #     feed_dict = {self.src_placeholder: sentence,
    #                  self.src_length_placeholder: np.array([length])}
        
    #     sentence_output = self.model.generate(self.session, feed_dict=feed_dict)
        
    #     if self.hparams.beam_width > 0:
    #         sentence_output = sentence_output[0]
    #     sep_token = self.hparams.sep_token.encode("utf-8")
    #     sentence_output = sentence_output.tolist()[0]
    #     if sep_token in sentence_output:
    #         sentence_output = sentence_output[:sentence_output.index(sep_token)]
    #     sentence_output = b' '.join(sentence_output).decode('utf-8')
        
    #     print(sentence_output)
        
    #     return sentence_output


if __name__ == "__main__":
    model = Interface(training = True)
#        print("# Start")
#        model = Interface(sess, training = False)
#        print("# Generate")
#        while True:
#            sentence = input("Q: ")
#            answer = model.generate(sentence, False)
#            print('-'*20)
#            print("A:", answer)
#            print('-'*20)
    
    