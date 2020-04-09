# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import meta_graph
from Bert import modeling
from nlp import Tokenizer
from hparams import HParams
from settings import SYSTEM_ROOT, BERT_PARAMS_FILE, BERT_CONFIG_FILE, VOCAB_FILE
from modules import get_initializer, check_vocab, variable_loader

class BertLoader():
    def __init__(self, hparams = None, tokenizer = None):
        self.hparams = hparams if hparams else HParams().hparams
        self.tokenizer = tokenizer if tokenizer else Tokenizer(hparams, VOCAB_FILE)
        
        self.bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            with tf.variable_scope('placeholder'):
                self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
                self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
                self.segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")
            
            self.model = modeling.BertModel(
                    config=self.bert_config,
                    is_training=False,
                    input_ids=self.input_ids,
                    input_mask=self.input_mask,
                    token_type_ids=self.segment_ids,
                    use_one_hot_embeddings=False,
                    scope="bert",
                    reuse=tf.AUTO_REUSE)
            
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, BERT_PARAMS_FILE)
            tf.train.init_from_checkpoint(BERT_PARAMS_FILE, assignment_map)
            
            self.variables = [v for v in tf.global_variables()]
            
            self.context_vector = tf.identity(self.model.get_sequence_output(), name="context_vector")
            self.finetuning_vector = tf.identity(self.model.get_pooled_output(), name="finetuning_vector")
    
    def connect_inputs(self, main_graph, input_ids, input_length, scope_name=None, name='bert_graph'):
        """Connect custom graph and inputs"""
        input_mask = tf.sequence_mask(input_length, tf.shape(input_ids)[1], dtype=tf.int32)
        segment_ids = tf.zeros_like(input_mask)
        scope_name = "{}/{}".format(main_graph.get_name_scope(), name) if main_graph.get_name_scope() else name
        sub_graph = tf.train.export_meta_graph(graph=self.graph)
        meta_graph.import_scoped_meta_graph(sub_graph, graph = main_graph,
                                            input_map={'placeholder/input_ids': input_ids,
                                                       'placeholder/input_mask': input_mask,
                                                       'placeholder/segment_ids': segment_ids}, 
                                            import_scope=name)
        self.context_vector = main_graph.get_tensor_by_name('{}/context_vector:0'.format(scope_name))
        self.finetuning_vector = main_graph.get_tensor_by_name('{}/finetuning_vector:0'.format(scope_name))
        variables_namelist = ["{}/{}".format(scope_name, v.name) for v in self.variables]
        return variables_namelist
    
    def get_bert_input(self, sentence_a, sentence_b=None, max_length=50):
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b) if sentence_b else None
        
        if len(tokens_a + tokens_b) > max_length:
            if tokens_b:
                tokens_b = tokens_b[:(max_length-len(tokens_a))-3]
            else:
                tokens_a = tokens_a[:max_length-2]
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"] if tokens_b else ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = np.zeros(len(tokens_a)+2, np.int32).tolist() + np.ones(len(tokens_b)+1, np.int32).tolist() if tokens_b else np.zeros(len(tokens), np.int32).tolist()

        inputs_length = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = np.ones(inputs_length, np.int32).tolist()
        
        if inputs_length < max_length:
            pad = np.zeros(max_length-inputs_length, np.int32).tolist()
            input_ids += pad
            input_mask += pad
            segment_ids += pad
        input_ids = np.reshape(np.array(input_ids), [1, -1])
        input_mask = np.reshape(np.array(input_mask), [1, -1])
        segment_ids = np.reshape(np.array(segment_ids), [1, -1])
        
        return input_ids, input_mask, segment_ids
#    
#    def get_context(self, session, text_a, text_b=None, max_length=50):
#        inp_ids, inp_mask, seg_ids = self.get_bert_input(text_a, text_b, max_length)
#        output = session.run(self.context_vector,feed_dict={self.input_ids: inp_ids, self.input_mask: inp_mask, self.segment_ids: seg_ids})
#        return output

if __name__ == "__main__":
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
    print(bert_config.hidden_size)
#    with tf.Session() as sess:
#        
#        hparams = HParams(SYSTEM_ROOT).hparams
#        
#        
#        graph = tf.get_default_graph()
#        
#    
#        with graph.as_default():
#            
#            bertloader = BertLoader(hparams = hparams)
#            
#            with tf.variable_scope('Network'):
#                
#                input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
#                input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
#                segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")
#                var = tf.Variable(3,name='var1')
#                
#            print(tf.global_variables())
##            split_saver = tf.train.Saver(tf.global_variables())
#            
#        variables_namelist = bertloader.connect_inputs(graph, input_ids, input_mask, segment_ids)
#    
##        tf.summary.FileWriter("tensorboard", graph)
#        
#        sess.run(tf.global_variables_initializer())
#        
#        var_list = [var for var in tf.global_variables() if var.name not in variables_namelist]
##        saver, _ = variable_loader(sess, RESULT_DIR, var_list=var_list)
#        print(sess.run(var))
#        
#        inp_ids, inp_mask, seg_ids = bertloader.get_bert_input("hello I am your partner .", "Hi ~ how are you ?", 50)
#        output = sess.run([bertloader.context_vector, bertloader.finetuning_vector],feed_dict={input_ids: inp_ids, input_mask: inp_mask, segment_ids: seg_ids})
#        print(output)
#        
##        saver.save(sess, RESULT_FILE, global_step = 100)
