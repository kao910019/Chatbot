# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.tensor_forest.python import tensor_forest

from Bert import modeling
from nlp import Tokenizer
from settings import VOCAB_FILE

from bert_loader import BertLoader
# original function in transformer
from modules import ln, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme, gelu
# record variable function
from modules import get_initializer, build_gradient, log_variable, avg_variable_list, summary_calculate, reward_plot, image_scalar, evaluate_classifier
# for update
from modules import build_gradient, focal_loss_softmax
# build network
from modules import build_rnn_encoder, build_liner_classifier, build_discriminator, build_rnn_cell
# useful function
from modules import check_nan, resort_zero, slice_multiterm_function, rebuild_mtem, padding_action, combine_bert_input, split_bert_output
from modules import regroup_sentence_one2one_form, one2one_form_summary, filter_complete_dialogue, get_memory_replay, get_sentence_length
class Transformer():
    def __init__(self, mode, graph, hparams, data_loader, batch_input):
        self.mode = mode
        self.training = False if mode == 'inference' else True
        
        self.graph = graph
        self.hparams = hparams
        
        self.enable_cluster = (self.hparams.job_name != None)
        self.vocab_dict = data_loader.vocab_dict
        self.vocab_size = data_loader.vocab_size
        self.vocab_table = data_loader.vocab_table
        self.reverse_vocab_table = data_loader.reverse_vocab_table
        
        self.emotion_list = data_loader.emotion_list
        self.emotion_size = data_loader.emotion_size
        self.emotion_table = data_loader.emotion_table
        self.reverse_emotion_table = data_loader.reverse_emotion_table
        
        self.dull_response_id = data_loader.dull_response_id
        
        self.batch_input = batch_input
        
        # This namelist can let you won't save the BERT variable, or that will occupy lots of memory.
        self.unsave_variable_namelist = []
        
        self.tokenizer = Tokenizer(hparams, VOCAB_FILE)
        # Memory replay
        self.memory = {'action':tuple(),'state':tuple(), 'rewards':tuple()}
        
        # Initializer
        initializer = get_initializer(self.hparams.init_op, self.hparams.random_seed, self.hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        with tf.variable_scope('Network'):
            with tf.variable_scope("vocab_embeddings", dtype = tf.float32):
                self.embedding = tf.get_variable("embedding", [self.vocab_size, self.hparams.num_dimensions], tf.float32)
            
            with tf.variable_scope('placeholder'):
                self.pad_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.pad_token)), tf.int32, name = 'pad_id')
                self.unk_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.unk_token)), tf.int32, name = 'unk_id')
                self.cls_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.cls_token)), tf.int32, name = 'cls_id')
                self.sep_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.sep_token)), tf.int32, name = 'sep_id')
                
                # Input eval
                self.source_id = tf.placeholder(dtype=tf.int32, shape=[None,None], name= "source_id")
                self.source_logits = tf.placeholder(dtype=tf.int32, shape=[None,None,None], name= "source_logits")
                self.batch_size = tf.shape(self.source_id, name= "batch_size")[0]
                self.half_batch_size = tf.cast(self.batch_size/2, tf.int32)
                self.source_length = get_sentence_length(self.source_id, self.batch_size, self.sep_id, self.hparams.max_length, name= "source_length")
                
                # Input target
                self.source_id_target = tf.identity(self.source_id, name= "source_id_target")
                self.source_logits_target = tf.identity(self.source_logits, name= "source_logits_target")
                self.source_length_target = get_sentence_length(self.source_id_target, self.batch_size_target, self.sep_id, self.hparams.max_length, name="source_length_target")
                self.batch_size_target = tf.reduce_sum(tf.minimum(self.source_length_target, 1))
                # Output target
                self.target_id_inputs = tf.identity(self.source_id, name= "target_id_inputs")
                self.target_id_outputs = tf.identity(self.source_id, name= "target_id_outputs")
                self.target_length = get_sentence_length(self.target_id_inputs, self.batch_size, self.sep_id, self.hparams.max_length, name="target_length")
                
                # Emotion
                self.source_emotion = tf.placeholder(dtype=tf.int32, shape=[None], name= "source_emotion")
                self.target_emotion = tf.identity(self.source_emotion, name= "target_emotion")
                
                # Control variable
                self.dropout_rate_flag = tf.Variable(False, name = 'dropout_rate_flag', trainable=False)
                self.dropout_rate = tf.cond(self.dropout_rate_flag, lambda: self.hparams.dropout_rate, lambda: 0.0, name='dropout_rate')
                self.passing_flag = tf.Variable(True, name = 'passing_flag', trainable=False)
                self.action_target_flag = tf.Variable(False, name = 'action_target_flag', trainable=False)
                self.response_flag = tf.Variable(False, name = 'response_flag', trainable=False)
                self.gan_train_flag = tf.Variable(False, name = 'gan_train_flag', trainable=False)
                self.term = tf.Variable(0, name = 'term', trainable=False)
                
                # Passing State
                self.latent_state_input = tf.placeholder(dtype=tf.float32, shape=[None,None,None], name='latent_state')
                self.latent_state_input_target = tf.identity(self.latent_state_input, name= "latent_state_target")
                
                # pretrain
                self.source_input = (self.source_id, self.source_length, self.source_emotion)
                self.target_input = (self.target_id_inputs, self.target_id_outputs, self.target_length, self.target_emotion)
                
                # RL
                self.k = tf.Variable(int(0.01* hparams.k_nearest*self.vocab_size), name = 'k', trainable=False)
                self.reward = tf.placeholder(dtype=tf.float32, shape=[None, None], name='reward')
                self.action_input_eval = (self.source_logits, self.source_id, self.source_length)
                self.action_input_target = (self.source_logits_target, self.source_id_target, self.source_length_target)
                self.accuracy_confusion = tf.Variable(1.0, name = 'accuracy_confusion', trainable=False)
                
                self.pretrain_epoch = tf.Variable(1, name = 'pretrain_epoch', trainable=False)
                self.classifier_epoch = tf.Variable(1, name = 'classifier_epoch', trainable=False)
                self.adversarial_epoch = tf.Variable(1, name = 'adversarial_epoch', trainable=False)
                self.ddpg_epoch = tf.Variable(1, name = 'ddpg_epoch', trainable=False)
                
                self.total_batch_size = tf.Variable(0, name = 'total_batch_size', trainable=False)
                self.assign_total_batch_size = tf.assign_add(self.total_batch_size, self.batch_size_target)
                self.reset_total_batch_size = tf.assign(self.total_batch_size, 0)
                
                # self.dataset_count_ph = tf.placeholder(dtype=tf.int32, shape=[], name= "dataset_count_ph")
                # self.dataset_count = tf.Variable(0, name = 'dataset_count', trainable=False)
                # self.assign_dataset_count = tf.assign(self.dataset_count, self.dataset_count_ph)
                
                self.global_step = tf.train.get_or_create_global_step()
        
            self.build_model(self.hparams)
        
            if self.training:
                self.build_train_model(self.hparams)
                self.build_special_summary()
                
                # Here is tensor that you want to log on summary for every epoch.
                with tf.variable_scope('Outputs'):
                    self.idle = tf.no_op(name="idle")
                    # avg_scale = tf.to_float(self.batch_size/self.dataset_count)
                    # self.pretrain_record_variable_sum = avg_variable_list(avg_scale, self.pretrain_record_variable)
                    # self.ddpg_record_variable_sum = avg_variable_list(avg_scale, self.ddpg_record_variable)
                    
                    self.term_reward = tf.placeholder(dtype=tf.string, shape=None)
                    self.term_reward_op = image_scalar("term_reward", self.term_reward)
                    
                    self.lambda_term_reward = tf.placeholder(dtype=tf.string, shape=None)
                    self.lambda_term_reward_op = image_scalar("lambda_term_reward", self.term_reward)
                    
                # Different mode update list make sure function_index divide (update step summary) <index> (log variable).
                if self.mode == 'pretrain':
                    self.step_summary = self.pretrain_summary
                    self.function_list = [self.pretrain_update, self.step_summary, self.assign_total_batch_size, self.passed_latent_state_eval]
                    self.update_list = self.function_list + self.pretrain_record_variable
                    self.update_function = self.pretrain_update_function
                    self.train_epoch = self.pretrain_epoch
                if self.mode == 'classifier':
                    self.step_summary = self.classifier_summary
                    self.function_list = [self.classifier_update, self.step_summary, self.assign_total_batch_size, self.passed_latent_state_eval]
                    self.update_list = self.function_list + self.classifier_record_variable
                    self.update_function = self.pretrain_update_function
                    self.train_epoch = self.classifier_epoch
                elif self.mode == 'adversarial':
                    self.step_summary = self.adversarial_summary
                    self.function_list = [self.adversarial_update, self.step_summary, self.assign_total_batch_size, self.passed_latent_state_eval]
                    self.update_list = self.function_list + self.adversarial_record_variable
                    self.update_function = self.adversarial_update_function
                    self.train_epoch = self.adversarial_epoch
                    
                self.assign_train_epoch = tf.assign_add(self.train_epoch, 1)
                self.funtion_index = len(self.function_list)
                
            # Create variable space for log epoch summary value.
            if self.training:
                if self.mode == 'pretrain':
                    log_variable(self.adversarial_record_variable, self.total_batch_size)
                    log_variable(self.classifier_record_variable, self.total_batch_size)
                    self.epoch_summary_op, self.epoch_summary_dict, self.log_variable_list = log_variable(self.pretrain_record_variable, self.total_batch_size)
                elif self.mode == 'classifier':
                    log_variable(self.adversarial_record_variable, self.total_batch_size)
                    self.epoch_summary_op, self.epoch_summary_dict, self.log_variable_list = log_variable(self.classifier_record_variable, self.total_batch_size)
                    log_variable(self.pretrain_record_variable, self.total_batch_size)
                elif self.mode == 'adversarial':
                    self.epoch_summary_op, self.epoch_summary_dict, self.log_variable_list = log_variable(self.adversarial_record_variable, self.total_batch_size)
                    log_variable(self.classifier_record_variable, self.total_batch_size)
                    log_variable(self.pretrain_record_variable, self.total_batch_size)
    
    def build_special_summary(self):
        with tf.variable_scope('summary'):
        # Summary
            self.one_batch_per_sec = tf.placeholder(dtype=tf.float32, shape=[], name='one_batch_per_sec')
            self.one_batch_per_sec_summary = tf.summary.scalar("one_batch_per_sec", self.one_batch_per_sec)
            self.unscaled_reward = tf.placeholder(dtype=tf.string, shape=None)
            self.scaled_reward = tf.placeholder(dtype=tf.string, shape=None)
            self.unscaled_reward_summary = image_scalar("unscaled_reward", self.unscaled_reward)
            self.scaled_reward_summary = image_scalar("scaled_reward", self.scaled_reward)
            self.summary_source_id = tf.placeholder(dtype=tf.int32, shape=[None,None], name= "summary_source_id")
            self.summary_target_id = tf.placeholder(dtype=tf.int32, shape=[None,None], name= "summary_target_id")
            self.summary_predict_id = tf.placeholder(dtype=tf.int32, shape=[None,None], name= "summary_predict_id")
        
            self.accuracy_confusion_iter = tf.assign(self.accuracy_confusion,  (0.97*self.accuracy_confusion + 0.01*self.discriminator_accuracy))
            self.train_GD_logit = tf.Variable(0, name = 'train_GD_logit_G0_D1', trainable=False)
            self.train_G_mode, self.train_D_mode = tf.assign(self.train_GD_logit, 0), tf.assign(self.train_GD_logit, 1)
            self.adversarial_record_variable += [self.accuracy_confusion, self.train_GD_logit]
            self.adversarial_summary = tf.summary.merge([tf.summary.scalar('step_adversarial_{}'.format(variable.op.name.split('/')[-1]), variable) for variable in self.adversarial_record_variable])
            target = True# if self.mode == 'pretrain' else False
            self.evaluation_summary = one2one_form_summary(
                self.hparams, self.reverse_vocab_table, self.summary_source_id, self.summary_target_id,  self.summary_predict_id, self.pad_id, target=target)
            self.multiturn_summary = tf.summary.merge([self.unscaled_reward_summary])
    
    def write_epoch_summary(self, session, summary_writer, epoch, options=None, run_metadata=None):
        feed_dict = {}
        for variable in self.log_variable_list:
            name = variable.op.name.split('/')[-1]
            feed_dict.update({variable: self.epoch_summary_dict[name]})
            self.epoch_summary_dict.update({name: 0.0})
        summary = session.run(self.epoch_summary_op, options=options, run_metadata=run_metadata, feed_dict=feed_dict)
        session.run([self.reset_total_batch_size], options=options, run_metadata=run_metadata, feed_dict=feed_dict)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()
    
    def write_step_summary(self, session, results, summary_writer, update=True):
        self.epoch_summary_dict = summary_calculate(self.epoch_summary_dict, self.log_variable_list, results[self.funtion_index:])
        if update:# and not self.enable_cluster:
            summary_writer.add_summary(results[1], session.run(self.global_step))
            summary_writer.flush()
    
    def copy_params_to_session(self, hparams, session1, session2):
        if (self.accuracy_confusion.eval(session=session2) - self.accuracy_confusion.eval(session=session1)) > hparams.accuarcy_copy_threshold:
            print("# Copy Session1 params to Session2 ...", end =" ")
            params = session1.run(tf.global_variables())
            copy_params = [tf.assign(t, e, name="copy_params") for t, e in zip(tf.global_variables(), params)]
            session2.run(copy_params)
            print("Done")
    
    def update(self, session1, session2, handler, writer=None, update = True, evaluation = False, options=None, run_metadata=None):
        # Get the batch
        batch_inputs = session1.run([self.batch_input.source, self.batch_input.target, self.batch_input.addition], 
                                   options=options, run_metadata=run_metadata, feed_dict=handler)
        input_list = self.update_list.copy()
        # Test Model
        input_list[0] = input_list[0] if update else self.idle
        return self.update_function(session1, session2, input_list, batch_inputs, writer=writer, update=update, evaluation=evaluation, options=options, run_metadata=run_metadata)
    
    def pretrain_update_function(self, session1, session2, input_list, batch_inputs, writer=None, update=True, evaluation=False, options=None, run_metadata=None):
        source, target, addition = batch_inputs
        batch_size, term_length = np.size(addition), addition
        term_latent_state = np.zeros([batch_size, self.hparams.num_rnn_layers, self.hparams.num_dimensions], np.float32)
        multiterm_source, multiterm_target, multiterm_predict = tuple(), tuple(), tuple()
        for index in range(term_length.max()):
            term_source, term_target = slice_multiterm_function(index, source, target)
            term_source, term_target = padding_action(self.hparams, term_source, ac_flag=False), padding_action(self.hparams, term_target, ac_flag=False)
            
            # source_flag, target_flag = np.minimum(term_source[1], 1), np.minimum(term_target[2], 1)
            # (term_source, term_target), (term_latent_state,) ,last_indice = \
            #     filter_complete_dialogue(source_flag, target_flag, (term_source, term_target), (term_latent_state,), last_indice)
            
            # if not source_flag.any() and  not target_flag.any():
            #     break
            
            results = session1.run(input_list, options=options, run_metadata=run_metadata, feed_dict={
                self.source_input: term_source, self.target_input: term_target, self.latent_state_input: term_latent_state, self.dropout_rate_flag: True})
            self.write_step_summary(session1, results, writer, update)
            term_latent_state = results[3]
            
            if evaluation:
                multiterm_source = rebuild_mtem(multiterm_source, (term_source[0],), axis=1)
                multiterm_target = rebuild_mtem(multiterm_target, (term_target[1],), axis=1)
                predict_id = session1.run(self.predict_id_eval, options=options, run_metadata=run_metadata, feed_dict={
                    self.source_input: term_source, self.target_input: term_target, self.latent_state_input: term_latent_state, self.dropout_rate_flag: True})
                multiterm_predict = rebuild_mtem(multiterm_predict, (predict_id,), axis=1)
                
        if evaluation:
            self.evaluation_multiterm_summary(session1, writer, multiterm_source, multiterm_predict, multiterm_target=multiterm_target, options=options, run_metadata=run_metadata)
        return session1.run(self.global_step)
    
    def adversarial_update_function(self, session1, session2, input_list, batch_inputs, writer=None, update=True, evaluation=False, options=None, run_metadata=None):
        source, target, addition = batch_inputs
        batch_size, term_length = np.size(addition), addition
        term_latent_state = np.zeros([batch_size, self.hparams.num_rnn_layers, self.hparams.num_dimensions], np.float32)
        for index in range(term_length.max()):
            term_source, term_target = slice_multiterm_function(index, source, target)
            term_source, term_target = padding_action(self.hparams, term_source, ac_flag=False), padding_action(self.hparams, term_target, ac_flag=False)
            
            self.adversarial_training(self.hparams, session1, term_source, term_target, term_latent_state, writer, evaluation=evaluation, options=options, run_metadata=run_metadata)
            
        return session1.run(self.global_step)
    
    def ddpg_update_function(self, session1, session2, input_list, batch_inputs, writer=None, update=True, evaluation=False, options=None, run_metadata=None):
        source, target, addition = batch_inputs
        batch_size, term_length = np.size(addition), addition
        if evaluation:
            multiterm_fw, multiterm_bw = tuple(), tuple()
            
        # QA Dialogue Dataset
        bw_inputs, fw_target = slice_multiterm_function(0, source, target)
        fw_pass_state_last = np.zeros([batch_size, self.hparams.num_rnn_layers, self.hparams.num_dimensions], np.float32)
        bw_pass_state_last = np.zeros([batch_size, self.hparams.num_rnn_layers, self.hparams.num_dimensions], np.float32)
        bw_action_last = (np.zeros(bw_inputs[0].shape + (self.vocab_size,)),) + bw_inputs[:-1]# id
        bw_action_last = padding_action(self.hparams, bw_action_last, ac_flag=False)
        fw_action_last = tuple((a*0.0).astype(a.dtype) for a in bw_action_last)
        plot_rewards = tuple()
        index = 0
        
#        def time_string(time):
#            time = round(time, 2)
#            return "{:02d}:{:02d}:{:02d}".format(int(time/3600), int(time%3600/60), int(time%60))
        
#        start_time = time.time()
        self.adversarial_training(self.hparams, session1, bw_inputs, fw_target, fw_pass_state_last, writer, evaluation=evaluation, options=options, run_metadata=run_metadata)
#        print("adversarial : {}".format(time_string(time.time() - start_time)))
#        self.copy_params_to_session(self.hparams, session1, session2)
        
        
        if False:#(self.accuracy_confusion.eval(session=session1) < self.hparams.discriminator_accuarcy_threshold):
            for index in range(self.hparams.max_term_length):

                
#                start_time = time.time()
#                print("forward start : ")
                fw_action, fw_pass_state, fw_response_flag = self.reinforcement_learning_env(session1, index, bw_action_last, fw_pass_state_last, policy=True, options=options, run_metadata=run_metadata)
#                print("forward : {}".format(time_string(time.time() - start_time)))
                
#                start_time = time.time()
#                print("backward start : ")
                bw_action, bw_pass_state, bw_response_flag = self.reinforcement_learning_env(session2, index, fw_action, bw_pass_state_last, policy=False, options=options, run_metadata=run_metadata)
#                print("backward : {}".format(time_string(time.time() - start_time)))
                
                if evaluation:
                    multiterm_bw = rebuild_mtem(multiterm_bw, (bw_action_last[1],), axis=1)
                    multiterm_fw = rebuild_mtem(multiterm_fw, (fw_action[1],), axis=1)
                
                if self.hparams.memory_replay:
#                    start_time = time.time()
                    rewards, plot_rewards = self.calculate_reward(session1, (bw_action_last, fw_action, fw_action_last), 
                                (fw_pass_state_last, bw_pass_state, bw_pass_state_last), plot_rewards, index, options=options, run_metadata=run_metadata)
#                    print("calculate reward : {}".format(time_string(time.time() - start_time)))
                    
                    (batch_fw_action, batch_bw_action_last, batch_bw_action), (batch_fw_pass_state_last, batch_fw_pass_state), batch_rewards, self.memory = \
                        get_memory_replay((fw_action, bw_action_last, bw_action), (fw_pass_state_last, fw_pass_state), rewards,
                            self.memory, self.hparams.batch_size, self.hparams.memory_capacity)
                    
                    feed_dict = {self.action_output_eval: batch_fw_action, self.action_input_eval: batch_bw_action_last, 
                                 self.action_input_target: batch_bw_action, self.latent_state_input: batch_fw_pass_state_last, 
                                 self.latent_state_input_target: batch_fw_pass_state, self.rewards:batch_rewards, self.action_target_flag: True}
                else:
                    feed_dict = {self.action_output_eval: fw_action, self.action_input_eval: bw_action_last, 
                                 self.action_input_target: bw_action, self.latent_state_input: fw_pass_state_last, 
                                 self.latent_state_input_target: fw_pass_state, self.rewards:rewards, self.action_target_flag: True}
                
                
#                start_time = time.time()
                results = session1.run(input_list, options=options, run_metadata=run_metadata, feed_dict=feed_dict)
#                print("update : {}".format(time_string(time.time() - start_time)))
#                sys.exit()
                
                
                self.write_step_summary(session1, results, writer, update)
                
                (fw_action, bw_action, fw_action_last, bw_action_last), (fw_pass_state, bw_pass_state, fw_pass_state_last, bw_pass_state_last), _ = \
                    filter_complete_dialogue(fw_response_flag, bw_response_flag, (fw_action, bw_action, fw_action_last, bw_action_last), 
                                             (fw_pass_state, bw_pass_state, fw_pass_state_last, bw_pass_state_last))
                if not fw_response_flag.any() and  not bw_response_flag.any():
                    break
                
                bw_action_last, bw_pass_state_last = bw_action, bw_pass_state
                fw_action_last, fw_pass_state_last = fw_action, fw_pass_state
            
            if index > 0:
                rewards_name = ["Emotional Expression", "Discriminator Score", "Spoken Interaction"]
                # rewards_name = ["Ease of Answering", "Information Flow", "Semantic Coherence", "Emotional Expression", "Discriminator Score", "Spoken Interaction"]
                term_length = plot_rewards[0].shape[0]
                unscaled_reward = reward_plot(plot_rewards, rewards_name, term_length, title='Unscaled Reward')
                # update model
                multiturn_summary = session1.run(self.multiturn_summary, options=options, run_metadata=run_metadata, 
                                      feed_dict={self.unscaled_reward: unscaled_reward})
                writer.add_summary(multiturn_summary, session1.run(self.global_step))
                
                if evaluation:
                    self.evaluation_multiterm_summary(session1, writer, multiterm_bw, multiterm_fw, multiterm_target=None, options=options, run_metadata=run_metadata)
        return session1.run(self.global_step)
    
    def evaluation_multiterm_summary(self, session, writer, multiterm_source, multiterm_predict, multiterm_target=None, options=None, run_metadata=None):
        multiterm_source = tuple(np.reshape(m,((-1,)+m.shape[2:])) for m in multiterm_source)
        multiterm_predict = tuple(np.reshape(m,((-1,)+m.shape[2:])) for m in multiterm_predict)
        sri = np.random.randint(0,multiterm_source[0].shape[0])
        mri = sri+self.hparams.max_term_length
        if multiterm_target is not None:
            multiterm_target = tuple(np.reshape(m,((-1,)+m.shape[2:])) for m in multiterm_target)
            eval_summary = session.run(self.evaluation_summary, options=options, run_metadata=run_metadata, feed_dict={
                self.summary_source_id: multiterm_source[0][sri:mri], self.summary_target_id: multiterm_target[0][sri:mri], self.summary_predict_id: multiterm_predict[0][sri:mri]})
        else:
            eval_summary = session.run(self.evaluation_summary, options=options, run_metadata=run_metadata, feed_dict={
                self.summary_source_id: multiterm_source[0][sri:mri], self.summary_predict_id: multiterm_predict[0][sri:mri]})
        writer.add_summary(eval_summary, session.run(self.global_step))

    def reinforcement_learning_env(self, session, index, action, last_state, policy=True, options=None, run_metadata=None):
        k = int(0.01* self.hparams.k_nearest*self.vocab_size) if policy else 0
        action, pass_state, response_flag = session.run([
            self.action_output_eval, self.passed_latent_state_eval, self.judgment_id_eval], options=options, run_metadata=run_metadata,  
            feed_dict = {self.action_input_eval : action, self.latent_state_input : last_state, self.k:k})
        action = padding_action(self.hparams, action)
        return action, pass_state, response_flag
    
    def calculate_reward(self, session, actions, states, plot_rewards, term, options=None, run_metadata=None):
        reward_action_input, new_plot_rewards = tuple(), tuple()
        rewards = []
        q_a, a_a, la_a = actions
        q_s, a_s, la_s = states
        def concat_action(actions):
            output = tuple()
            for a1, a2 in zip(*actions):
                output += (np.concatenate((a1, a2), axis=0),)
            return output
        def concat_state(states):
            return np.concatenate(states, axis=0)
        
        rewards = session.run([self.emotional_expression, self.discriminator_score, self.spoken_interaction], options=options, run_metadata=run_metadata,
                              feed_dict={self.action_input_eval: q_a, self.latent_state_input: q_s, self.term: term})
        first = False
        if plot_rewards == tuple():
            first, plot_rewards = True, tuple(np.mean(np.reshape(r, [1,-1]), axis=1) for r in rewards)
        for plot_reward, reward in zip(plot_rewards, rewards):
            reward = np.concatenate([plot_reward, np.mean(np.reshape(reward, [1,-1]), axis=1)], axis=0) if not first else plot_reward
            new_plot_rewards += (reward,)
        return rewards, new_plot_rewards
    
    def adversarial_training(self, hparams, session, source, target, state, writer, evaluation=False, options=None, run_metadata=None):
        if hparams.split_train_gd:
            if ((self.accuracy_confusion.eval(session=session) > self.hparams.train_generator_threshold) and self.train_GD_logit.eval(session=session) == 1):
                session.run(self.train_G_mode)
            elif((self.accuracy_confusion.eval(session=session) < self.hparams.train_discriminator_threshold) and self.train_GD_logit.eval(session=session) == 0):
                session.run(self.train_D_mode)
        
        source, target = padding_action(hparams, source, ac_flag=False), padding_action(hparams, target, ac_flag=False)
        
        if hparams.split_train_gd:
            if self.train_GD_logit.eval(session=session) == 0:
                discriminator_accuracy, predict_id, _, summary = session.run([self.discriminator_accuracy, self.predict_id_eval, self.g_update, self.adversarial_summary], options=options, run_metadata=run_metadata,
                    feed_dict={self.source_input: source, self.target_input: target, self.latent_state_input: state, self.response_flag:True, self.gan_train_flag:True})
            else:
                discriminator_accuracy, predict_id, _, summary = session.run([self.discriminator_accuracy, self.predict_id_eval, self.d_update, self.adversarial_summary], options=options, run_metadata=run_metadata,
                    feed_dict={self.source_input: source, self.target_input: target, self.latent_state_input: state, self.response_flag:True, self.gan_train_flag:True})
        else:
            discriminator_accuracy, predict_id, _, summary = session.run([self.discriminator_accuracy, self.predict_id_eval, self.adversarial_update, self.adversarial_summary], options=options, run_metadata=run_metadata,
                feed_dict={self.source_input: source, self.target_input: target, self.latent_state_input: state, self.response_flag:True, self.gan_train_flag:True})
       
        writer.add_summary(summary, session.run(self.global_step))
        
        if evaluation:
            self.evaluation_multiterm_summary(session, writer, rebuild_mtem(tuple(), (source[0],), axis=1), rebuild_mtem(tuple(), (predict_id,), axis=1), 
                                    multiterm_target= rebuild_mtem(tuple(), (target[1],), axis=1), options=options, run_metadata=run_metadata)
        session.run(self.accuracy_confusion_iter, feed_dict= {self.discriminator_accuracy: discriminator_accuracy})
        
    def generate(self, session, feed_dict):
        session.run(self.batch_input.initializer, feed_dict=feed_dict)
        feed_dict.update({self.batch_inputs_placeholder: (self.batch_input.source, self.batch_input.target, self.batch_input.batch_size)})
        predict_sentence = session.run([self.predict_sentence], feed_dict=feed_dict)
        print(predict_sentence)
        return predict_sentence
    
    def build_train_model(self, hparams):
        """
            Unsupervised Learning
        """
        with tf.variable_scope("pretrain"):
            warmup_learning_rate = noam_scheme(hparams.init_learning_rate, self.global_step, hparams.warmup_steps)
            warmup_learning_rate = tf.identity(warmup_learning_rate, name= "warmup_learning_rate")
            # Emotion Classifier
            pointer_target_weight = tf.cast(tf.minimum(self.target_emotion, 1), tf.float32)
            pointer_target_count = tf.reduce_sum(pointer_target_weight)
#            pointer_loss = tf.reduce_mean(focal_loss_softmax(labels = self.target_emotion, logits = self.pointer_logits_eval, gamma=3) * pointer_target_weight, name='pointer_loss')
            pointer_loss = tf.divide(focal_loss_softmax(labels = self.target_emotion, logits = self.pointer_logits_eval, gamma=3) * pointer_target_weight, pointer_target_count, name='pointer_loss')
            
            emotion_label = tf.concat([self.target_emotion, self.source_emotion], axis=0)
            emotion_target_weight = tf.cast(tf.minimum(emotion_label, 1), tf.float32)
            emotion_target_count = tf.reduce_sum(emotion_target_weight)
#            emotion_loss = tf.reduce_mean(focal_loss_softmax(labels = emotion_label, logits = self.emotion_logits, gamma=3) * emotion_target_weight, name='emotion_loss')
            emotion_loss = tf.divide(focal_loss_softmax(labels = emotion_label, logits = self.emotion_logits, gamma=3) * emotion_target_weight, emotion_target_count, name='emotion_loss')
            
            pointer_evaluate = evaluate_classifier(labels=self.target_emotion, logits=self.pointer_logits_eval, num_class=self.emotion_size, num_pad=1, name_list=self.emotion_list, name="pointer")
            emotion_evaluate = evaluate_classifier(labels=emotion_label, logits=self.emotion_logits, num_class=self.emotion_size, num_pad=1, name_list=self.emotion_list, name="emotion")
            
#            hparams = tensor_forest.ForestHParams(num_classes=self.emotion_size, num_features=hparams.num_dimensions, num_trees=20, max_nodes=1000).fill()
#            forest_graph = tensor_forest.RandomForestGraphs(hparams)
#            # Get training graph and loss
#            train_op = forest_graph.training_graph(self.emotion_logits, emotion_label)
#            loss_op = forest_graph.training_loss(self.emotion_logits, emotion_label)
#            infer_op, _, _ = forest_graph.inference_graph(self.emotion_logits)
#            correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(emotion_label, tf.int64))
#            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            if hparams.train_encoder_by_classifier:
                pointer_update = build_gradient(hparams, pointer_loss, self.encoder_eval_params + self.pointer_eval_params, learning_rate=2e-5)
            else:
                pointer_update = build_gradient(hparams, pointer_loss, self.pointer_eval_params, learning_rate=2e-5)
            
            global_step = self.global_step if self.mode == 'classifier' else None 
            emotion_update = build_gradient(hparams, emotion_loss, self.emotion_params, step = self.global_step, learning_rate=2e-5)
            
            # Discriminator
            target_count = tf.reduce_sum(self.reward_target)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reward_labels, logits=self.reward_logits)
            discriminator_loss = tf.divide(tf.reduce_sum(crossent * self.reward_target), tf.maximum(target_count, 1.0), name='discriminator_loss')
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reward_labels, logits=self.reward_tf_logits)
            tf_discriminator_loss = tf.divide(tf.reduce_sum(crossent * self.reward_target), tf.maximum(target_count, 1.0), name='tf_discriminator_loss')
            # discriminator_loss = tf.reduce_mean(tf.reduce_sum(crossent * self.reward_target, axis=-1), name='discriminator_loss')
            self.discriminator_accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.reward_logits, axis=-1, output_type=tf.int32), self.reward_labels), tf.float32) * self.reward_target), tf.maximum(target_count, 1.0), name='discriminator_accuracy')
            self.tf_discriminator_accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.reward_tf_logits, axis=-1, output_type=tf.int32), self.reward_labels), tf.float32) * self.reward_target), tf.maximum(target_count, 1.0), name='tf_discriminator_accuracy')
            
            # discriminator_accuracy = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.reward_logits, axis=-1, output_type=tf.int32), self.reward_labels), tf.float32) * self.reward_target, axis=-1), name='discriminator_accuracy')
            
            if hparams.dynamic_discriminator_learning_rate:
                discriminator_learning_rate = hparams.init_learning_rate * 4e-3 * tf.pow(5.0, (1/tf.pow(self.accuracy_confusion,2.0)))
            else:
                discriminator_learning_rate = hparams.init_learning_rate
            discriminator_learning_rate = tf.identity(discriminator_learning_rate, name= "discriminator_learning_rate")
            
            global_step = self.global_step if hparams.split_train_gd else None
            discriminator_update = build_gradient(hparams, discriminator_loss, self.discriminator_params, step = global_step, learning_rate=discriminator_learning_rate)
            tf_discriminator_update = build_gradient(hparams, tf_discriminator_loss, self.tf_discriminator_params, learning_rate=discriminator_learning_rate*0.1)
            
            # Response judgment
            judgment_labels = tf.minimum(self.target_length, 1)
            judgment_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=judgment_labels, logits=self.judgment_logits_eval), name='judgment_loss')
            judgment_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.judgment_logits_eval, axis=1), tf.int32), judgment_labels), tf.float32), name='judgment_accuracy')
            
            response_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=judgment_labels, logits=self.response_logits), name='response_loss')
            response_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.response_logits, axis=1), tf.int32), judgment_labels), tf.float32), name='response_accuracy')
            
            if hparams.train_encoder_by_classifier:
                judgment_update = build_gradient(hparams, judgment_loss, self.encoder_eval_params + self.judgment_eval_params)
            else:
                judgment_update = build_gradient(hparams, judgment_loss, self.judgment_eval_params)
                
            response_update = build_gradient(hparams, response_loss, self.response_params)
            
            # Transformer
            max_time = tf.shape(self.target_id_outputs)[1]
            target_weight = tf.sequence_mask(self.target_length, max_time, dtype=tf.float32)
            
            if hparams.label_smooth:
                labels = label_smoothing(tf.one_hot(self.target_id_outputs, depth=self.vocab_size))
                crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.decode_logits_eval)
            else:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_id_outputs, logits=self.decode_logits_eval)
                
            # my version
            target_count = tf.reduce_sum(target_weight, axis=-1)
            target_batch = tf.reduce_sum(tf.minimum(target_count, 1.0), axis=-1)
            transformer_loss = tf.divide(tf.reduce_sum(tf.reduce_sum(crossent * target_weight, axis=-1) / tf.maximum(target_count, 1e-7)), tf.maximum(target_batch, 1.0), name='transformer_loss')
            
            # normal word average
#            transformer_loss = tf.divide(tf.reduce_sum(crossent * target_weight), tf.maximum(tf.reduce_sum(target_weight), 1e-7), name='transformer_loss')
            
            # sentence acerage
#            transformer_sentence_loss = tf.reduce_mean(tf.reduce_sum(crossent * target_weight, axis=-1), name='transformer_sentence_loss')
            
            transformer_perplexity = tf.exp(transformer_loss, name='transformer_perplexity')
            
            transformer_update = build_gradient(hparams, transformer_loss, (self.encoder_eval_params + self.transformer_eval_params + self.latent_eval_params),
                                                step = self.global_step, learning_rate = warmup_learning_rate)
            
            self.pretrain_update = [emotion_update, pointer_update, discriminator_update, tf_discriminator_update, judgment_update, response_update, transformer_update]
            self.pretrain_record_variable = [warmup_learning_rate, emotion_loss, pointer_loss, discriminator_loss, self.discriminator_accuracy, self.tf_discriminator_accuracy, response_loss, response_accuracy, judgment_loss, judgment_accuracy,
                                             transformer_loss, transformer_perplexity] + emotion_evaluate + pointer_evaluate
            self.pretrain_summary = tf.summary.merge([tf.summary.scalar('step_{}'.format(variable.op.name.split('/')[-1]), variable) for variable in self.pretrain_record_variable])
            
            self.classifier_update = [emotion_update, discriminator_update, tf_discriminator_update, response_update]
            self.classifier_record_variable = [warmup_learning_rate, emotion_loss, discriminator_loss, self.discriminator_accuracy, self.tf_discriminator_accuracy, response_loss, response_accuracy] + emotion_evaluate
            self.classifier_summary = tf.summary.merge([tf.summary.scalar('step_{}'.format(variable.op.name.split('/')[-1]), variable) for variable in self.classifier_record_variable])
            
            
        """
            Reinforcement Learning
                Deep Deterministic Policy Gradient (DDPG)
        """
        with tf.name_scope('dialogue_reward'):
            # Emotional Expression [Q] [bs]
            emotion_scale = 1.0
            emotion_logits = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.emotion_logits, lambda: self.emotion_logits[:self.batch_size])
            cosine = tf.keras.losses.cosine_similarity(self.pointer_logits_eval, emotion_logits)
            self.emotional_expression = tf.negative(tf.log(tf.sigmoid(cosine)) * emotion_scale, name='emotional_expression')
            
            # Discriminator Score [Q] [bs, len]
            reward_scale = 1.0
            reward_target = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.reward_target, lambda: self.reward_target[:self.batch_size])
            reward_logits = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.reward_logits, lambda: self.reward_logits[:self.batch_size])
            reward_tf_logits = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.reward_tf_logits, lambda: self.reward_tf_logits[:self.batch_size])
                
            target_count = tf.reduce_sum(reward_target, axis=-1)
            self.discriminator_score = tf.identity(tf.subtract(tf.nn.softmax(reward_logits)[:,:,1], 0.5) * reward_scale, name='discriminator_score')
            self.tf_discriminator_score = tf.identity(tf.subtract(tf.nn.softmax(reward_tf_logits)[:,:,1], 0.5) * reward_scale, name='tf_discriminator_score')
            
            # self.discriminator_score = tf.divide(tf.reduce_sum(reward * self.reward_target[:self.third_batch_size], axis=-1), tf.maximum(target_count, 1.0), name='discriminator_score')
            # self.discriminator_score = tf.reduce_mean(reward * self.reward_target[:self.third_batch_size], axis=-1, name='discriminator_score')
        
            # Spoken Interaction [Q] [bs]
            interaction_scale = 0.5 * tf.cast(hparams.max_term_length-self.term, tf.float32)
            cosine = tf.keras.losses.cosine_similarity(self.response_logits, self.judgment_logits_eval)
            self.spoken_interaction = tf.negative(tf.log(tf.sigmoid(cosine)) * interaction_scale, name='spoken_interaction')
            
            self.rewards = (self.emotional_expression, self.discriminator_score, self.spoken_interaction)
        
        with tf.variable_scope('adversarial', reuse=tf.AUTO_REUSE):
            emotional_expression = tf.reduce_mean(self.emotional_expression, name="reward_emotional_expression")
            discriminator_score = tf.reduce_mean(self.discriminator_score, name="reward_discriminator_score")
            tf_discriminator_score = tf.reduce_mean(self.tf_discriminator_score, name="reward_tf_discriminator_score")
            spoken_interaction = tf.reduce_mean(self.spoken_interaction, name="reward_spoken_interaction")
            
            max_time = tf.shape(self.predict_id_eval)[1]
            target_weight = tf.sequence_mask(self.predict_length_eval, max_time, dtype=tf.float32)
            last_target = target_weight - tf.sequence_mask(self.predict_length_eval-1, 9, dtype=tf.float32)
#            target_count = tf.reduce_sum(target_weight, axis=1)
            
            reward = tf.zeros_like(target_weight)
            if hparams.emotion_reward:
                reward += last_target * tf.reshape(self.emotional_expression, [-1,1])
            if hparams.response_reward:
                reward += last_target * tf.reshape(self.spoken_interaction, [-1,1])
            target_count = tf.reduce_sum(target_weight)
            reward = (self.discriminator_score + self.discriminator_score) * target_weight
            
            if hparams.discount_reward:
                reward = tf.transpose(tf.scan(lambda a, x: a*hparams.gamma + x, tf.transpose(reward), reverse=True))
                if hparams.normalize_reward:
    #                reward -= tf.reshape((tf.reduce_sum(reward, axis=-1)/target_count),[-1,1])
                    reward -= tf.reshape(tf.reduce_mean(reward, axis=-1),[-1,1])
                    reward /= tf.reshape(tf.math.reduce_std(reward, axis=-1),[-1,1])
            else:
                reward = -reward
            
            total_reward = tf.reduce_mean(reward, name= "discriminator_learning_rate")
                
            if hparams.label_smooth:
                labels = label_smoothing(tf.one_hot(self.predict_id_eval, depth=self.vocab_size))
                crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.predict_logits_eval)
            else:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.predict_id_eval, logits=self.predict_logits_eval)
                
            target_count = tf.reduce_sum(target_weight, axis=-1)
            target_batch = tf.reduce_sum(tf.minimum(target_count, 1.0), axis=-1)
            adversarial_loss = tf.divide(tf.reduce_sum(tf.reduce_sum(crossent * reward * target_weight, axis=-1) / tf.maximum(target_count, 1e-7)), tf.maximum(target_batch, 1.0), name='adversarial_loss')
            
            self.adversarial_update = build_gradient(hparams, adversarial_loss, (self.encoder_eval_params + self.transformer_eval_params + self.latent_eval_params), step=self.global_step, learning_rate = warmup_learning_rate)
            
            with tf.control_dependencies([self.adversarial_update]):
                self.teacher_forcing_update = build_gradient(hparams, transformer_loss, (self.encoder_eval_params + self.pointer_eval_params + self.transformer_eval_params + self.latent_eval_params), learning_rate = warmup_learning_rate)
            
            self.d_update = [discriminator_update, tf_discriminator_update, emotion_update, response_update]
            self.g_update = [self.adversarial_update, self.teacher_forcing_update, pointer_update, judgment_update, emotion_update, response_update]
            self.adversarial_update = [self.adversarial_update, self.teacher_forcing_update, emotion_update, pointer_update, judgment_update, response_update, discriminator_update, tf_discriminator_update]
            self.adversarial_record_variable = emotion_evaluate + pointer_evaluate + [transformer_loss, transformer_perplexity, adversarial_loss, emotion_loss, discriminator_loss, self.discriminator_accuracy, self.tf_discriminator_accuracy, discriminator_learning_rate, warmup_learning_rate,
                                                                                      emotional_expression, discriminator_score, tf_discriminator_score, spoken_interaction, total_reward]
            
            
    def build_model(self, hparams):
        # actor eval
        with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
            self.input_context_eval, self.input_vector_eval, self.output_context_eval, self.output_vector_eval, \
            self.passed_latent_state_eval, self.pointer_logits_eval, self.judgment_logits_eval, self.judgment_id_eval, \
            self.decode_logits_eval, self.decode_id_eval, self.decode_length_eval, self.predict_logits_eval, self.predict_id_eval, self.predict_length_eval, \
            self.encoder_eval_params, self.latent_eval_params, self.pointer_eval_params, self.judgment_eval_params, self.transformer_eval_params, self.actor_eval_params = \
                self.build_actor(hparams, self.latent_state_input, self.passing_flag, self.source_id, self.source_length, 
                                 self.target_id_inputs, self.target_id_outputs, self.target_length, self.response_flag, self.dropout_rate, name='eval', trainable=True)
            self.action_output_eval = (self.predict_logits_eval, self.predict_id_eval, self.predict_length_eval)
            
        # actor target
        if hparams.target_network:
            with tf.variable_scope('Actor', reuse=tf.AUTO_REUSE):
                self.input_context_target, self.input_vector_target, self.output_context_target, self.output_vector_target, \
                self.passed_latent_state_target, self.pointer_logits_target, self.judgment_logits_target, self.judgment_id_target, \
                self.decode_logits_target, self.decode_id_target, self.decode_length_target, self.predict_logits_target, self.predict_id_target, self.predict_length_target, \
                self.encoder_target_params, self.latent_target_params, self.pointer_target_params, self.response_target_params, self.transformer_target_params, self.actor_target_params = \
                    self.build_actor(hparams, self.latent_state_input_target, self.passing_flag, self.source_id_target, self.source_length_target, 
                                     self.target_id_inputs, self.target_id_outputs, self.target_length, self.response_flag, self.dropout_rate, name='target', trainable=False)
                self.action_output_target = (self.predict_logits_target, self.predict_id_target, self.predict_length_target)
                
        # ---------------------------------------------------------------
        # Classifier and Discriminator
        # ---------------------------------------------------------------
        with tf.variable_scope('Emotion_Classifier', reuse=tf.AUTO_REUSE) as scope:
            predict_id_inputs = tf.concat([tf.expand_dims(self.hparams.cls_id * tf.cast(tf.not_equal(self.predict_length_eval, 0), tf.int32), -1), self.predict_id_eval], axis=-1)[:,:self.hparams.max_length]
#            predict_length_inputs = tf.minimum(tf.where(tf.not_equal(self.predict_length_eval, 0), self.predict_length_eval+1, self.predict_length_eval), self.hparams.max_length)
            
            if self.mode in ['pretrain', 'classifier']:
                inputs_id = tf.concat([self.target_id_inputs, self.source_id], axis=0) 
            else:
                inputs_id = tf.cond(tf.equal(self.gan_train_flag, False), lambda: predict_id_inputs,  lambda: tf.concat([predict_id_inputs, self.source_id], axis=0))
            
            encode_context = self.build_transformer_encoder(hparams, inputs_id, self.dropout_rate)
            self.emotion_logits = build_liner_classifier(hparams, encode_context[:,0,:], self.emotion_size)
            self.emotion_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        
        with tf.variable_scope('Response_Classifier', reuse=tf.AUTO_REUSE) as scope:
            encode_context = self.build_transformer_encoder(hparams, self.source_id, self.dropout_rate)
            self.response_logits = build_liner_classifier(hparams, encode_context[:,0,:], 2)
            self.response_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        
        with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE) as scope:
            if self.mode in ['pretrain', 'classifier']:
                source_id = tf.concat([self.source_id, self.source_id], axis=0)
                source_length = tf.concat([self.source_length, self.source_length], axis=0)
                target_id = tf.concat([self.decode_id_eval, self.target_id_outputs], axis=0)
                target_length = tf.concat([self.decode_length_eval, self.target_length], axis=0)
            else:
                source_id = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.source_id, lambda: tf.concat([self.source_id, self.source_id], axis=0))
                source_length = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.source_length, lambda: tf.concat([self.source_length, self.source_length], axis=0))
                target_id = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.predict_id_eval, lambda: tf.concat([self.predict_id_eval, self.target_id_outputs], axis=0))
                target_length = tf.cond(tf.equal(self.gan_train_flag, False), lambda: self.predict_length_eval, lambda: tf.concat([self.predict_length_eval, self.target_length], axis=0))
            with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE) as scope:
                self.reward_labels, self.reward_target, self.reward_logits = build_discriminator(hparams, self.embedding, source_id, source_length, target_id, target_length, self.dropout_rate)
                self.discriminator_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            with tf.variable_scope('Transformer', reuse=tf.AUTO_REUSE) as scope:
                self.reward_tf_logits = self.build_transformer_discriminator(hparams, source_id, source_length, target_id, target_length)
                self.tf_discriminator_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
    
    def build_transformer_discriminator(self, hparams, source_ids, source_length, target_ids, target_length):
        pad_id = 0
        discriminator_inputs = tf.concat([source_ids, target_ids], axis= 1)
        batch_size, max_time = tf.shape(discriminator_inputs)[0], tf.shape(discriminator_inputs)[1]
            
        odd_indices = tf.matmul(tf.reshape(tf.range(1, batch_size*2, 2), [-1,1]), tf.fill([1,max_time], 1))
        even_indices = tf.matmul(tf.reshape(tf.range(0, batch_size*2, 2), [-1,1]), tf.fill([1,max_time], 1))
        partitions = tf.where(tf.not_equal(discriminator_inputs, pad_id), even_indices, odd_indices)
        sequence_parts = tf.dynamic_partition(discriminator_inputs, partitions, hparams.batch_size*4)
        discriminator_inputs = tf.reshape(tf.concat(sequence_parts, axis=0), tf.shape(discriminator_inputs))
        # Create input_mask and segment_ids.
        target_mask = tf.sequence_mask(source_length+target_length, max_time)
        segment_ids = tf.cast(tf.logical_xor(tf.sequence_mask(source_length, max_time), target_mask), tf.int32)
        
        encode_context = self.build_transformer_encoder(hparams, discriminator_inputs, self.dropout_rate, max_length=hparams.max_length*2)
        logits = build_liner_classifier(hparams, encode_context, 2)
        
        labels = segment_ids
        odd_indices = tf.matmul(tf.reshape(tf.range(1, batch_size*2, 2), [-1,1]), tf.fill([1, tf.shape(labels)[1]], 1))
        even_indices = tf.matmul(tf.reshape(tf.range(0, batch_size*2, 2), [-1,1]), tf.fill([1, tf.shape(labels)[1]], 1))
        partitions = tf.where(tf.not_equal(labels, 0), even_indices, odd_indices)
        sequence_parts = tf.dynamic_partition(logits, partitions, hparams.batch_size*4)
        logits = tf.reshape(tf.concat(sequence_parts, axis=0), tf.shape(logits))[:,:tf.shape(target_ids)[1]]
        return logits
    
    
    def build_latent_encoder(self, hparams, vector_input, initial_state, dropout_rate=0.0, passing_flag=None, cell=None, trainable=True, emotion_pointer=True):
        _, latent_state, latent_cell = build_rnn_encoder(
            hparams, tf.expand_dims(vector_input, axis=1), initial_state=initial_state, passing_flag=passing_flag, cell=cell, input_layer=False,
            dropout_rate=dropout_rate, trainable=trainable, scope='latent_encoder')
        
        with tf.variable_scope('Emotion_Pointer', reuse=tf.AUTO_REUSE) as pointer_scope:
            inputs = tf.reshape(latent_state, [-1, hparams.num_rnn_layers*hparams.num_dimensions])
            logits =  tf.layers.dense(ln(inputs), hparams.num_rnn_layers*hparams.num_dimensions, activation=gelu, trainable=trainable)
            pointer_logits = tf.layers.dense(logits, self.emotion_size, use_bias=False, trainable=trainable)
            emotion_pointer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=pointer_scope.name)
            
        if emotion_pointer:
            latent_state += tf.reshape(logits, [-1, hparams.num_rnn_layers, hparams.num_dimensions])
            
        latent_state = ln(latent_state)
        return latent_state, latent_cell, pointer_logits, emotion_pointer_params
    
    def build_actor(self, hparams, initial_state, passing_flag, source_id, source_length, target_id_inputs, target_id_outputs, target_length, 
                    response_flag, dropout_rate, name='eval', trainable=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as main_scope:
            # Latent Encoder <===> Emotion Pointer
            with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
                if hparams.use_rnn_encoder:
                    encoder_outputs, encoder_state, encoder_cell = build_rnn_encoder(hparams, source_id, embedding = self.embedding, length=source_length, cell=None)
                    input_context, input_vector = encoder_outputs, tf.reshape(encoder_state, [-1, hparams.num_rnn_layers*hparams.num_dimensions])
                else:
                    input_context = self.build_transformer_encoder(hparams, source_id, dropout_rate)
                    input_vector = input_context[:,0,:]
            
            with tf.variable_scope('Latent_Encoder', reuse=tf.AUTO_REUSE):
                latent_state, latent_cell, pointer_logits, _ = \
                    self.build_latent_encoder(hparams, input_vector, initial_state, dropout_rate=dropout_rate, passing_flag=passing_flag, cell=None, trainable=trainable, emotion_pointer=True)
                inputs = latent_state[:,1,:]
                
            with tf.variable_scope('Response_Judgment', reuse=tf.AUTO_REUSE) as scope:
                judgment_logits = build_liner_classifier(hparams, inputs, 2, trainable=trainable)
                judgment_id = tf.minimum(target_length, 1) if (self.mode in ['pretrain', 'classifier', 'adversarial']) else tf.cast(tf.argmax(judgment_logits, axis=1), tf.int32)
                judgment_id = tf.cond(response_flag, lambda: tf.ones_like(judgment_id), lambda: judgment_id)
                judgment_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                
            with tf.variable_scope('Transformer', reuse=tf.AUTO_REUSE) as scope:
                if hparams.use_rnn_encoder:
                    decode_logits, decode_id, decoder_length, predict_logits, predict_id, predict_length = \
                        self.build_rnn_decoder(hparams, input_context, latent_state, source_id, target_id_inputs, judgment_id, dropout_rate, training=self.training, trainable=trainable)   
                else:
                    memory = input_context + tf.tile(tf.expand_dims(inputs, axis=1), [1, tf.shape(input_context)[1], 1])
                    memory = ln(input_context)
                    decode_logits, decode_id, decode_length, _, _ = \
                        self.build_transformer_decoder(hparams, memory, source_id, target_id_inputs, None, None, None, judgment_id, dropout_rate, training=self.training, trainable=trainable)
                    predict_logits, predict_id, predict_length = \
                        self.build_evaluator_transformer(hparams, memory, source_id, source_length, judgment_id, dropout_rate, trainable=trainable)
                    
                transformer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                
            with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
                id_inputs = target_id_outputs if (self.mode in ['pretrain', 'classifier']) else predict_id
                length_inputs = target_length if (self.mode in ['pretrain', 'classifier']) else predict_length
                
                id_inputs = tf.concat([tf.expand_dims(hparams.cls_id * tf.cast(tf.not_equal(length_inputs, 0), tf.int32), -1), id_inputs], axis=-1)[:,:hparams.max_length]
                length_inputs = tf.minimum(tf.where(tf.not_equal(length_inputs, 0), length_inputs+1, length_inputs), hparams.max_length)
                
                if hparams.use_rnn_encoder:
                    encoder_outputs, encoder_state, _ = build_rnn_encoder(id_inputs, embedding = self.embedding, length=length_inputs, cell=encoder_cell)
                    output_context, output_vector = encoder_outputs, tf.reshape(encoder_state, [-1, hparams.num_rnn_layers*hparams.num_dimensions])
                else:
                    output_context = self.build_transformer_encoder(hparams, id_inputs, dropout_rate)
                    output_vector = output_context[:,0,:]
                encoder_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            
            with tf.variable_scope('Latent_Encoder', reuse=tf.AUTO_REUSE) as scope:
                passed_latent_state, _, _, emotion_pointer_params = \
                    self.build_latent_encoder(hparams, output_vector, latent_state, dropout_rate=dropout_rate, passing_flag=None, cell=latent_cell, trainable=trainable, emotion_pointer=False)
                latent_encoder_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                latent_encoder_params = [var for var in latent_encoder_params if var not in emotion_pointer_params]
                
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=main_scope.name)
#            params = [var for var in params if var not in critic_params]
            
        return input_context, input_vector, output_context, output_vector, \
            passed_latent_state, pointer_logits, judgment_logits, judgment_id, \
            decode_logits, decode_id, decode_length, predict_logits, predict_id, predict_length, \
            encoder_params, latent_encoder_params, emotion_pointer_params, judgment_params, transformer_params, params
    
    def build_critic(self, hparams, encoder_state, source_id, source_length, action_id, action_length, dropout_rate, source_state=None, trainable=True, while_loop=False, policy=False):
        if not while_loop:
            encoder_state = tf.layers.dense(encoder_state, hparams.num_dimensions, trainable=trainable, name="encoder_state_dense")
            source_length = get_sentence_length(source_id, tf.shape(source_id)[0], self.sep_id, hparams.max_length)
            _, source_state, _ = build_rnn_encoder(hparams, source_id, embedding=self.embedding, length=source_length, dropout_rate=dropout_rate, trainable=trainable, scope="source_rnn")
            source_state = tf.concat([tf.tile(tf.expand_dims(encoder_state, axis=1), [1, hparams.num_rnn_layers, 1]), source_state], axis=2)
            source_state = tf.layers.dense(source_state, hparams.num_dimensions, activation=gelu, name="source_state_dense")
            
        if while_loop:
            if policy:
                # From paper : Deep Reinforcement Learning in Large Discrete Action Spaces - Wolpertinger Policy
                with tf.name_scope('Wolpertinger_Policy'):
                    batch_size, input_max_time, last_ids = tf.shape(action_id)[0], tf.shape(action_id)[1], tf.reshape(action_id[:,-1], [-1, 1])
                    cosine_similarity = tf.matmul(tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embedding, tf.reshape(last_ids, [-1])), dim=1), 
                                                  tf.transpose(tf.nn.l2_normalize(self.embedding, dim=1)))
                    _, similar_id = tf.nn.top_k(cosine_similarity, k=self.k)
                    input_action_id = action_id
                    action_id = tf.reshape(tf.concat([last_ids, similar_id], axis=1), [-1, 1])
                    
                    source_state = tf.contrib.seq2seq.tile_batch(source_state, self.k+1)
                    action_length = tf.minimum(get_sentence_length(action_id, batch_size*(self.k+1), self.sep_id, hparams.max_length), 1)
                    
            else:
                batch_size = tf.shape(action_id)[0]
                action_id = tf.reshape(action_id[:,-1], [-1, 1])
                action_length = tf.minimum(get_sentence_length(action_id, batch_size, self.sep_id, hparams.max_length), 1)
        
        action_output, action_state, _ = build_rnn_encoder(hparams, action_id, initial_state=source_state, embedding=self.embedding, length=action_length, dropout_rate=dropout_rate, trainable=trainable)
        
        max_time = tf.shape(action_output)[1]
        q = tf.reshape(tf.layers.dense(action_output, 1, trainable=trainable), [-1, max_time])
        
        if while_loop and policy:
            with tf.name_scope('Wolpertinger_Policy'):
                q = tf.reshape(tf.transpose(q), [1, batch_size, self.k+1])
                argmax_q_indice = tf.reshape(tf.argmax(q, axis=2), [1, batch_size, 1])
                q = tf.transpose(tf.gather_nd(q, argmax_q_indice, batch_dims=2))
                action_id = tf.reshape(tf.transpose(action_id), [1, batch_size, self.k+1])
                action_id = tf.transpose(tf.gather_nd(action_id, argmax_q_indice, batch_dims=2))
                
                action_id = tf.concat([input_action_id[:,:input_max_time-1], action_id], axis=1)
                action_length = get_sentence_length(action_id, batch_size, self.sep_id, hparams.max_length)
                
                action_state = tf.reshape(action_state, [1, batch_size, self.k+1, hparams.num_rnn_layers, hparams.num_dimensions])
                action_state = tf.gather_nd(action_state, argmax_q_indice, batch_dims=2)[0]
        return action_id, action_length, q, action_state
    
    def build_evaluator_transformer(self, hparams, memory, source_id, source_length, judgment_id, dropout_rate, trainable=True):
        batch_size = tf.shape(source_id)[0]
        
        with tf.name_scope('evaluator_variable'):
            # input logits ids length
            init_loop_decoder_logits = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True, clear_after_read=True) 
            init_loop_decoder_id = tf.TensorArray(dtype=tf.int32, infer_shape=False, size=1, dynamic_size=True, clear_after_read=False) 
            init_loop_decoder_logits = init_loop_decoder_logits.write(0, tf.zeros([batch_size, self.vocab_size], tf.float32))
            init_loop_decoder_id = init_loop_decoder_id.write(0, tf.zeros([batch_size], tf.int32))
            # Attention K & V cache
            init_loop_self_cache = [tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True, clear_after_read=False) for i in range(hparams.num_blocks)]
            init_loop_self_cache = [init_loop_self_cache[i].write(
                    0, [tf.zeros([batch_size*hparams.num_blocks, int(hparams.num_dimensions/hparams.num_blocks)], np.float32), 
                        tf.zeros([batch_size*hparams.num_blocks, int(hparams.num_dimensions/hparams.num_blocks)], np.float32)]) for i in range(hparams.num_blocks)]
            init_loop_source_cache = [
                    [tf.zeros([batch_size*hparams.num_blocks, hparams.max_length, int(hparams.num_dimensions/hparams.num_blocks)], np.float32),
                     tf.zeros([batch_size*hparams.num_blocks, hparams.max_length, int(hparams.num_dimensions/hparams.num_blocks)], np.float32)] for i in range(hparams.num_blocks)]
            
        def decode_loop(stop_flag, step, while_loop, loop_self_cache, loop_source_cache, loop_decoder_logits, loop_decoder_id):
            input_id = tf.concat([tf.fill([batch_size, 1], self.cls_id), tf.transpose(loop_decoder_id.stack()[1:])], axis = 1)
#            self_cache = [loop_self_cache[i].read(step-1) for i in range(hparams.num_blocks)]
            self_cache = [tf.transpose(loop_self_cache[i].stack()[1:], [1, 2, 0, 3]) for i in range(hparams.num_blocks)]
            
            logits, ids, length, self_cache, loop_source_cache = self.build_transformer_decoder(hparams, memory, source_id, input_id, self_cache, loop_source_cache, while_loop, judgment_id, dropout_rate, training=False, trainable=trainable)
            
            loop_self_cache = [loop_self_cache[i].write(step, self_cache[i][:, :,-1,:]) for i in range(hparams.num_blocks)]
            loop_decoder_logits = loop_decoder_logits.write(step, logits[:,-1,:])
            loop_decoder_id = loop_decoder_id.write(step, ids[:,-1])
            
            stop_flag = tf.logical_or(tf.greater_equal(step, hparams.max_length), 
                                      tf.reduce_all(tf.reduce_any(tf.logical_or(tf.equal(ids, self.sep_id), tf.equal(ids, self.pad_id)), 1), 0))
            return stop_flag, tf.add(step, 1), while_loop, loop_self_cache, loop_source_cache, loop_decoder_logits, loop_decoder_id
        
        # First Loop
        stop_flag, step, _, init_loop_self_cache, init_loop_source_cache, init_loop_decoder_logits, init_loop_decoder_id = decode_loop(
                False, 1, None, init_loop_self_cache, init_loop_source_cache, init_loop_decoder_logits, init_loop_decoder_id)
        
        _, step, _, _, _, decoder_logits, decoder_id\
          = tf.while_loop(lambda stop_flag, *_: tf.logical_not(stop_flag), decode_loop,
                          loop_vars=[stop_flag, step, True, init_loop_self_cache, init_loop_source_cache, init_loop_decoder_logits, init_loop_decoder_id],
                          back_prop = True, name='decode_loop')
          
        decoder_logits = tf.transpose(decoder_logits.stack()[1:], [1,0,2])
        decoder_id = tf.transpose(decoder_id.stack()[1:])
        
        length = tf.shape(decoder_id)[1]
        decoder_logits = tf.concat([decoder_logits, tf.fill([batch_size, hparams.max_length-length, self.vocab_size], 0.0)], axis=1)
        decoder_id = tf.concat([decoder_id, tf.fill([batch_size, hparams.max_length-length], 0)], axis=1)
        decoder_length = get_sentence_length(decoder_id, batch_size, self.sep_id, hparams.max_length)
        return decoder_logits, decoder_id, decoder_length
    
    def build_transformer_encoder(self, hparams, encoder_inputs, dropout_rate, training=True, max_length=None):
        max_length = hparams.max_length if max_length == None else max_length
        # src_masks
        src_masks = tf.math.equal(encoder_inputs, 0) # (N, T1)
        # embedding
        enc = tf.nn.embedding_lookup(self.embedding, encoder_inputs) # (N, T1, d_model)
        enc *= hparams.num_dimensions**0.5 # scale
        enc += positional_encoding(enc, max_length)
        enc = tf.layers.dropout(enc, dropout_rate, training=training)
        ## Blocks
        for i in range(hparams.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc, _, _, _ = multihead_attention(
                    queries=enc, keys=enc, values=enc, key_masks=src_masks,
                    cache=None, memory_cache=False, while_loop=None, share_softmax=None, share_attention=None, num_heads=hparams.num_heads,
                    dropout_rate=dropout_rate, training=training, causality=False)
                # feed forward
                enc = ff(enc, num_units=[hparams.num_ff_dimensions, hparams.num_dimensions])
        return enc
    
    def build_rnn_decoder(self, hparams, memory, latent_state, encoder_inputs, decoder_inputs,
                      judgment_id=None, dropout_rate=0.0, training=True, trainable=True):
        
        with tf.variable_scope("output_projection", reuse=tf.AUTO_REUSE):
            output_layer = layers_core.Dense(self.vocab_size, use_bias=False, name="output_projection")
            
        source_length =  get_sentence_length(decoder_inputs, tf.shape(decoder_inputs)[0], self.sep_id, hparams.max_length)
        cell, init_state = self.build_decoder_cell(hparams, source_length, memory, latent_state)
        
        with tf.name_scope("teacher_forcing_decoder"):
            decoder_inputs_length = get_sentence_length(decoder_inputs, tf.shape(decoder_inputs)[0], self.sep_id, hparams.max_length)
            decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, decoder_inputs)
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_inputs_length)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, init_state, output_layer = output_layer)
            decoder_outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory = True)
            
            decode_logits = decoder_outputs.rnn_output
            decode_id = decoder_outputs.sample_id
            decoder_length = get_sentence_length(decode_id, tf.shape(decode_id)[0], self.sep_id, hparams.max_length)
            
        with tf.name_scope("sample_decoder"):
            helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding, start_tokens=self.cls_id, end_token=self.sep_id)
            decoder_sample = tf.contrib.seq2seq.BasicDecoder(cell, helper_sample, init_state, output_layer = output_layer)
            output_sample, output_sample_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder_sample, swap_memory=True, maximum_iterations = hparams.max_length)
        
            predict_logits = output_sample.rnn_output
            predict_id = output_sample.sample_id
            if judgment_id != None:
                predict_id *= tf.reshape(judgment_id, [-1 , 1])
            predict_length = get_sentence_length(predict_id, tf.shape(predict_id)[0], self.sep_id, hparams.max_length)
            
        return decode_logits, decode_id, decoder_length, predict_logits, predict_id, predict_length
        
    def build_decoder_cell(self, hparams, source_length, memory, latent_state):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        if not self.training and hparams.beam_width > 0:
            with tf.name_scope('Tile_batch'):
                source_length = tf.contrib.seq2seq.tile_batch(source_length, multiplier=hparams.beam_width)
                memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=hparams.beam_width)
                latent_state = tf.contrib.seq2seq.tile_batch(latent_state, multiplier=hparams.beam_width)
            beam_size = self.batch_size * hparams.beam_width
        else:
            beam_size = self.batch_size
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            hparams.num_dimensions, memory, memory_sequence_length=source_length)
        cell = build_rnn_cell(hparams, self.dropout_rate, trainable=True)
        # Only generate alignment in greedy INFER mode.
        alignment_history = (not self.training and hparams.beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=hparams.num_dimensions, alignment_history=alignment_history, name="attention")
        init_state = cell.zero_state(beam_size, tf.float32).clone(cell_state=latent_state)
        return cell, init_state
    
    def build_transformer_decoder(self, hparams, memory, encoder_inputs, decoder_inputs, self_cache=None, source_cache=None, while_loop=None,
                      judgment_id=None, dropout_rate=0.0, training=True, trainable=True):
        # src_masks
        src_masks = tf.math.equal(encoder_inputs, 0) # (N, T1)
        # tgt_masks
        tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)
        # embedding
        dec = tf.nn.embedding_lookup(self.embedding, decoder_inputs)  # (N, T2, d_model)
        dec *= hparams.num_dimensions ** 0.5  # scale
        dec += positional_encoding(dec, hparams.max_length)
        dec = tf.layers.dropout(dec, dropout_rate, training=training)
        # Blocks
        for i in range(hparams.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # Masked self-attention (Note that causality is True at this time)
                if (i == 0 or i%hparams.share_attention==1):
                    softmax, attention = None, None
                
                self_c = None if self_cache is None else self_cache[i]
                source_c = None if self_cache is None else source_cache[i]
                
                dec, self_c, softmax, _ = multihead_attention(
                    queries=dec, keys=dec, values=dec, key_masks=tgt_masks, 
                    cache=self_c, memory_cache=False, while_loop=while_loop, share_softmax=softmax, share_attention=None, num_heads=hparams.num_heads,
                    dropout_rate=dropout_rate, training=training, causality=True, scope="self_attention", trainable=trainable)
                # Vanilla attention
                dec, source_c, _, attention = multihead_attention(
                    queries=dec, keys=memory, values=memory, key_masks=src_masks, 
                    cache=source_c, memory_cache=True, while_loop=while_loop, share_softmax=None, share_attention=attention, num_heads=hparams.num_heads,
                    dropout_rate=dropout_rate, training=training, causality=False, scope="vanilla_attention", trainable=trainable)
                
#                dec, self_c, softmax, _ = multihead_attention(
#                    queries=dec, keys=dec, values=dec, key_masks=tgt_masks, 
#                    cache=None, memory_cache=False, while_loop=None, share_softmax=None, share_attention=None, num_heads=hparams.num_heads,
#                    dropout_rate=dropout_rate, training=training, causality=True, scope="self_attention", trainable=trainable)
#                # Vanilla attention
#                dec, source_c, _, attention = multihead_attention(
#                    queries=dec, keys=memory, values=memory, key_masks=src_masks, 
#                    cache=None, memory_cache=False, while_loop=None, share_softmax=None, share_attention=None, num_heads=hparams.num_heads,
#                    dropout_rate=dropout_rate, training=training, causality=False, scope="vanilla_attention", trainable=trainable)
                
                
                if self_cache is not None:
                    self_cache[i] = self_c
                if source_cache is not None:
                    source_cache[i] = source_c
                
                ### Feed Forward
                dec = ff(dec, num_units=[hparams.num_ff_dimensions, hparams.num_dimensions], trainable=trainable)
        
        weights = tf.transpose(self.embedding) # (d_model, vocab_size)
        if hparams.output_activation  == 'leaky_relu':
            logits = tf.nn.leaky_relu(tf.einsum('ntd,dk->ntk', dec, weights)) # (N, T2, vocab_size)
        elif hparams.output_activation  == 'gelu':
            logits = gelu(tf.einsum('ntd,dk->ntk', dec, weights)) # (N, T2, vocab_size)
        else:
            logits = tf.einsum('ntd,dk->ntk', dec, weights)
        
        batch_size, max_length = tf.shape(dec)[0], tf.shape(dec)[1]
        
        ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
        # Response Judgment flag
        if judgment_id != None:
            ids *= tf.reshape(judgment_id, [-1 , 1])
            # first_id = ids[:,0]
            # first_id = tf.reshape(tf.where(tf.equal(first_id, 0), tf.ones_like(first_id) * self.sep_id, first_id), [-1, 1])
            # ids = tf.concat([first_id , ids[:,1:]], axis=-1)
        # Detect PAD and clear behind words
        length = get_sentence_length(ids, batch_size, self.sep_id, hparams.max_length)
        target_mask = tf.cast(tf.sequence_mask(length, max_length), tf.int32)
        ids *= target_mask
        return logits, ids, length, self_cache, source_cache
    
    