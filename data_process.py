# -*- coding: utf-8 -*-
import os
import sys
import copy

from tqdm import tqdm
from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from hparams import HParams
from settings import SYSTEM_ROOT, VOCAB_FILE, EMOTION_FILE, RECORD_FILE_NAME_LIST, SPLIT_LIST, RECORD_DIR, CORPUS_DATA_DIR
from settings import CORPUS_DATA_MTEM_DIR, CORPUS_DATA_QA_DIR
from settings import DULL_RESPONSE

from modules import check_vocab, save_vocab, rebuild_vocab

from nlp import Tokenizer, Relation_Extraction

ELEMENT_LIST = [{'name':'senin_id', 'dtype':np.int64, 'out_dtype':np.int32},
                {'name':'senout_id', 'dtype':np.int64, 'out_dtype':np.int32},
                {'name':'senin_length', 'dtype':np.int64, 'out_dtype':np.int32},
                {'name':"senout_length", 'dtype':np.int64, 'out_dtype':np.int32},
                {'name':"emoin_id", 'dtype':np.int64, 'out_dtype':np.int32},
                {'name':"emoout_id", 'dtype':np.int64, 'out_dtype':np.int32}]

class DataLoader(object):
    def __init__(self, hparams, tokenizer=None, training=True, mode='inference'):
        
        self.training = training
        self.hparams = hparams
        
        self.tokenizer = tokenizer if tokenizer else Tokenizer(self.hparams, VOCAB_FILE)
        self.vocab_size, self.vocab_dict = len(self.tokenizer.vocab), self.tokenizer.vocab
        
        self.emotion_tokenizer = tokenizer if tokenizer else Tokenizer(self.hparams, EMOTION_FILE)
        self.emotion_size, self.emotion_list = len(self.emotion_tokenizer.vocab), self.emotion_tokenizer.inv_vocab
        
        with tf.name_scope("data_process"):
        
            self.vocab_table = lookup_ops.index_table_from_file(
                    VOCAB_FILE, default_value=self.hparams.unk_id)
            self.reverse_vocab_table = lookup_ops.index_to_string_table_from_file(
                    VOCAB_FILE, default_value=self.hparams.unk_token)
            
            self.emotion_table = lookup_ops.index_table_from_file(
                    EMOTION_FILE, default_value=self.hparams.unk_id)
            self.reverse_emotion_table = lookup_ops.index_to_string_table_from_file(
                    EMOTION_FILE, default_value=self.hparams.unk_token)
            
            self.dull_response_id = self.get_dull_response(DULL_RESPONSE)
        
        
        if self.training:
            with tf.name_scope("load_record"):
                if mode == 'ddpg':
                    train_file = 'daily_train.tfrecords'
                    test_file = 'daily_test.tfrecords'
                else:
                    train_file = 'daily_mtem_train.tfrecords'
                    test_file = 'daily_mtem_test.tfrecords'
#                    train_file = 'friends_train.tfrecords'
#                    test_file = 'friends_test.tfrecords'
                self.train_dataset_count, self.train_dataset = self.load_record(os.path.join(RECORD_DIR,train_file), ELEMENT_LIST)
                self.test_dataset_count, self.test_dataset = self.load_record(os.path.join(RECORD_DIR,test_file), ELEMENT_LIST)
    
    def get_dull_response(self, dull_response):
        dull_response_id = []
        max_length = 0
        for response in dull_response:
            tokens = self.tokenizer.tokenize(response)
            tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
            max_length = len(tokens_id) if max_length < len(tokens_id) else max_length
        for response in dull_response:
            tokens = self.tokenizer.tokenize(response)
            tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
            length = len(tokens_id)
            tokens_id = tokens_id + [self.hparams.sep_id] + [self.hparams.pad_id for _ in range(max_length - length)]
            dull_response_id.append(tokens_id)
        dull_response_id = tf.convert_to_tensor(dull_response_id)
        return dull_response_id
        
    def get_emotion_num(self, batch_lists, tag='emoin'):
        emotion_num_dict = dict()
        for batch_list in batch_lists:
            for batch_dict in batch_list:
                emo_tag = '_{}_'.format(batch_dict[tag])
                if emo_tag not in emotion_num_dict:
                    emotion_num_dict[emo_tag] = 1
                else:
                    emotion_num_dict[emo_tag] += 1
        return emotion_num_dict
    
    def get_emotion_weight(self, emotion_num_dict):
        emotion_weight_dict = dict()
        total_sum, total_mul = 0.0, 1.0
        for index in range(1, self.emotion_size):
            total_sum += emotion_num_dict[self.emotion_list[index]]
#            total_mul *= emotion_num_dict[self.emotion_list[index]]
        for index in range(1, self.emotion_size):
#            emotion_weight_dict.update({self.emotion_list[index]: (1.0 / emotion_num_dict[self.emotion_list[index]]) * 100.0})
            emotion_weight_dict.update({self.emotion_list[index]: (total_sum - emotion_num_dict[self.emotion_list[index]]) / total_sum})
        emotion_weight_dict.update({self.emotion_list[0]: 0.0})
        return emotion_weight_dict
    
    def parse_example(self, serial_exmp, element_list):
        sequence_features = {}
        for element in element_list:
            sequence_features.update({element['name']: tf.FixedLenSequenceFeature([], dtype=element['dtype'])})
        context, feature_lists = tf.parse_single_sequence_example(
                serialized=serial_exmp,sequence_features=sequence_features)
        outputs = []
        for element in element_list:
            outputs.append(tf.cast(feature_lists[element['name']], element['out_dtype']))
        return tuple(outputs)
    
    def load_record(self, load_file, element_list):
        count = 0
        for record in tf.python_io.tf_record_iterator(load_file):
            count += 1
        dataset = tf.data.TFRecordDataset(load_file)
        dataset = dataset.map(lambda serial_exmp: self.parse_example(serial_exmp, element_list))
        return count, dataset
    
    def get_training_batch(self, input_dataset):
        with tf.name_scope("batch_dataset"):
            buffer_size = self.hparams.batch_size * 400
            
            # Reshape MTEM dialogue to 2 dimension
            dataset = input_dataset.map(lambda si, so, il, ol, ei, eo: \
                (tf.reshape(si, [tf.size(il), -1]), tf.reshape(so, [tf.size(ol), -1]),
                  il, ol, ei, eo, tf.size(il))).prefetch(buffer_size)
                
            dataset = dataset.map(lambda si, so, il, ol, ei, eo, tl: \
                (tf.concat([tf.expand_dims(self.hparams.cls_id * tf.cast(tf.not_equal(il, 0), tf.int32), -1), si], axis=-1),
                 tf.concat([tf.expand_dims(self.hparams.cls_id * tf.cast(tf.not_equal(ol, 0), tf.int32), -1), so], axis=-1), so, 
                 tf.where(tf.not_equal(il, 0), il+1, il), ol, ei, eo, tl)).prefetch(buffer_size)
            
            # Limit sentence length
            dataset = dataset.map(lambda si, ti, to, il, ol, ei, eo, tl: \
                (si[:,:self.hparams.max_length],
                 ti[:,:self.hparams.max_length], to[:,:self.hparams.max_length], 
                 tf.minimum(il, self.hparams.max_length), tf.minimum(ol, self.hparams.max_length), ei, eo, tl)).prefetch(buffer_size)
            
            shuffle_dataset = dataset.shuffle(buffer_size=buffer_size)
            
            batched_dataset = shuffle_dataset.padded_batch(self.hparams.batch_size,
                        padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                                       tf.TensorShape([None]), tf.TensorShape([None]),
                                       tf.TensorShape([None]), tf.TensorShape([None]),
                                       tf.TensorShape([])),
                        padding_values=(self.hparams.pad_id, self.hparams.pad_id, self.hparams.pad_id, # sentence
                                        0, 0, # sentence length
                                        self.hparams.pad_id, self.hparams.pad_id, # sentence emotion
                                        0))
            
            iterator = batched_dataset.make_initializable_iterator()
            encoder_input, decoder_input, decoder_output, encoder_length, decoder_length, encoder_emotion, decoder_emotion, term_length = iterator.get_next()
        
        return BatchedInput(iterator=iterator, batched_dataset=batched_dataset, handle=iterator.string_handle(), initializer=iterator.initializer,
                            source=(encoder_input, encoder_length, encoder_emotion), 
                            target=(decoder_input, decoder_output, decoder_length, decoder_emotion),
                            addition=term_length)
    
    def multiple_batch(self, handler, dataset):
        """Make Iterator switch to change batch input"""
        iterator = tf.data.Iterator.from_string_handle(handler, dataset.output_types, dataset.output_shapes)
        encoder_input, decoder_input, decoder_output, encoder_length, decoder_length, encoder_emotion, decoder_emotion, term_length = iterator.get_next()
        return BatchedInput(iterator=None, batched_dataset=None, handle=None, initializer=None,
                            source=(encoder_input, encoder_length, encoder_emotion), 
                            target=(decoder_input, decoder_output, decoder_length, decoder_emotion),
                            addition=term_length)
    
def QA_openfile(file, EM=False):
    """
    openfile with Q & A format.
    """
    batch_list = []
    with open(file, 'r', encoding = 'utf8', errors='ignore') as f:
        Q, A = "", ""
        for line in f:
            if line.startswith("Q: "):
                Q = line[3:]
            elif line.startswith("A: "):
                A = line[3:]
            if EM:
                if line.startswith("QE: "):
                    QE = line[4:]
                elif line.startswith("AE: "):
                    AE = line[4:]
            else:
                QE = hparams.pad_token
                AE = hparams.pad_token
                
            if Q != "" and A != "":
                batch_list.append([dict({'senin':Q, 'emoin':QE, 'senout':A, 'emoout':AE})])
                Q, A = "", ""
    return batch_list

def MTEM_openfile(file):
#     """
#     openfile with MTEM format.
#     """
    df = pd.read_json(file, orient='records')
    indexs, terms = np.shape(df)
    batch_list = []
    
    for index in range(indexs):
        sentence_list = []
        speaker_list = []
        for term in range(terms):
            try:
                if df[term][index] is not None:
                    if df[term][index]['speaker'] not in speaker_list:
                        speaker_list.append(df[term][index]['speaker'])
                    sentence_list.append({'speaker':df[term][index]['speaker'], 'utterance':df[term][index]['utterance'],
                                          'emotion':df[term][index]['emotion'], 'annotation':df[term][index]['annotation']})
            except:
                pass
        for speaker in speaker_list:
            output_list=[]
            
            sen_input, emo_predict, sen_output, emo_output = hparams.pad_token, hparams.pad_token, hparams.pad_token, hparams.pad_token
            Q_term = False
            for index, sentence in enumerate(sentence_list):
                if Q_term and sentence['speaker'] != speaker:
                    output_list.append(dict({'senin':sen_input, 'emoin':emo_predict, 'senout':sen_output, 'emoout':emo_output}))
                    sen_input, emo_predict, sen_output, emo_output = hparams.pad_token, hparams.pad_token, hparams.pad_token, hparams.pad_token
                    Q_term = False
                if not Q_term and sentence['speaker'] != speaker:
                    sen_input, emo_predict = sentence['utterance'], sentence['emotion']
                    Q_term = True
                elif sentence['speaker'] == speaker:
                    output_list.append(dict({'senin':sen_input, 'emoin':emo_predict, 'senout':sentence['utterance'], 'emoout':sentence['emotion']}))
                    sen_input, emo_predict, sen_output, emo_output = hparams.pad_token, hparams.pad_token, hparams.pad_token, hparams.pad_token
                    Q_term = False
                else:
                    output_list.append(dict({'senin':sen_input, 'emoin':emo_predict, 'senout':sen_output, 'emoout':emo_output}))
                    sen_input, emo_predict, sen_output, emo_output = hparams.pad_token, hparams.pad_token, hparams.pad_token, hparams.pad_token
                    Q_term = False
            if Q_term:
                output_list.append(dict({'senin':sen_input, 'emoin':emo_predict, 'senout':sen_output, 'emoout':emo_output}))
                sen_input, emo_predict, sen_output, emo_output = hparams.pad_token, hparams.pad_token, hparams.pad_token, hparams.pad_token
                Q_term = False
            batch_list.append(output_list)
    return batch_list

def save_dataset(element_list, file = 'trainingdata'):
    writer = tf.python_io.TFRecordWriter(os.path.join(RECORD_DIR,"{}.tfrecords".format(file)))
    
    for values in zip(*[element['values'] for element in element_list]):
        feature_dict = {}
        for element, value in zip(element_list, values):
            # Notice that v in value must have one dimension : Shape(:,)
            if element['dtype'] == np.int64:
                try:
                    data = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(v, np.int64)])) for v in value]
                except:
                    data = tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(value, np.int64)]))
            feature_dict.update({element['name']: tf.train.FeatureList(feature=data)})
        feature_list = tf.train.FeatureLists(feature_list=feature_dict)
        example = tf.train.SequenceExample(feature_lists=feature_list)
        writer.write(example.SerializeToString())
    writer.close()
    
def search_id(hparams, tokenizer, batch_list, target, emo_target):
    batch_id_list = []
    batch_emo_list = []
    batch_length_list = []
    
    emotion_size, emotion_list = check_vocab(EMOTION_FILE)
    
    for terms in tqdm(batch_list, desc="search_id : {}".format(target)):
        sentence_id_list = []
        sentence_emo_list = []
        sentence_length_list = []
        max_length = 0
        for sentence in terms:
            if '[unused' in sentence[target].strip():
                sents = sentence[target].strip().split(' ')
                tokens_id = []
                for word in sents:
                    token_id = tokenizer.convert_tokens_to_ids_single(word)
                    tokens_id += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) if token_id == hparams.unk_id else [token_id]
            else:
                tokens = tokenizer.tokenize(sentence[target].strip()) if sentence[target].strip() != hparams.pad_token else [hparams.pad_token]
                tokens_id = tokenizer.convert_tokens_to_ids(tokens)
                
            if tokens_id[0] == hparams.pad_id:
                # tokens_id = [hparams.sep_id]
                tokens_id = []
            else:
                tokens_id.append(hparams.sep_id)
                
            sen_length = len(tokens_id)
            # sen_length = 0 if tokens_id[0] == hparams.pad_id else sen_length
            max_length = sen_length if max_length < sen_length else max_length
            
            emo_id = emotion_list.index(sentence[emo_target].strip())
            
            sentence_id_list.append(tokens_id)
            sentence_emo_list.append(emo_id)
            sentence_length_list.append(sen_length)
            
        first = True
        for sentence_id in sentence_id_list:
            if len(sentence_id) < max_length:
                pad_sentence_id = np.append(sentence_id, hparams.pad_id * np.zeros((max_length - len(sentence_id)), np.int64))
            else:
                pad_sentence_id = np.array(sentence_id, np.int64)
            if first:
                first = False
                new_sentence_id_list = np.array(pad_sentence_id, np.int64)
            else:
                new_sentence_id_list = np.append(new_sentence_id_list, pad_sentence_id)
                
        batch_id_list.append(new_sentence_id_list)
        batch_emo_list.append(sentence_emo_list)
        batch_length_list.append(sentence_length_list)
    return batch_id_list, batch_emo_list, batch_length_list

class BatchedInput(namedtuple("BatchedInput",
                              ["iterator","batched_dataset","handle","initializer",
                               "source","target","addition"])):
    pass

def relation_process(extracter, batch, force=False):
    new_batch = []
    for sentence_list in tqdm(batch, desc="relation_process"):
        if force:
            new_batch.append(copy.deepcopy(sentence_list))
            sentence_list[0]['senout'] = copy.deepcopy(sentence_list[0]['senin'])
        if sentence_list[0]['senin'] == hparams.pad_token or force:
            sentence_list[0]['senin'] = extracter.build_triples_sentence(extracter.get_triples(copy.deepcopy(sentence_list[0]['senin'])))
            sentence_list[0]['emoout'] = copy.deepcopy(sentence_list[0]['emoin'])
            new_batch.append(copy.deepcopy(sentence_list))
    return new_batch

def generate_dataset(hparams, corpus_dir, dialogue_type = 'QA', types = 'txt', conbine = True):
    
#    extracter = Relation_Extraction()
    tokenizer = Tokenizer(hparams, VOCAB_FILE)
    
    # Walk through directory
    file_list = []
    for path, dirs, files in sorted(os.walk(corpus_dir)):
            if files:
                for file in files:
                    file_path = os.path.join(path, file)
                    if file.lower().endswith('.{}'.format(types)):
                        file_list.append(file_path)
    # Read file
    batch_list = []
    for file in file_list:
        if dialogue_type == 'QA':
            batch = QA_openfile(file, EM=True)
#            batch = relation_process(extracter, batch, force=True)
        elif dialogue_type == 'MTEM':
            batch = MTEM_openfile(file)
#            batch = relation_process(extracter, batch)
        if conbine:
            batch_list += batch
        else:
            batch_list.append(batch)
    file_batch_list = [batch_list] if conbine else batch_list
    
    # Generate tfrecord
    for file, batch_list in zip(file_list, file_batch_list):
        file_name = file.split('\\')[-1][:-(len(types)+1)]
        print("File : {}, Batch_size : {}".format(file_name, len(batch_list)))
        senin_id, emoin_id, senin_length = search_id(hparams, tokenizer, batch_list, 'senin', 'emoin')
        senout_id, emoout_id, senout_length = search_id(hparams, tokenizer, batch_list, 'senout', 'emoout')
        element_values = (senin_id, senout_id, senin_length, senout_length, emoin_id, emoout_id)
        for element, values in zip(ELEMENT_LIST, element_values):
            element.update({'values':values})
        file_name = 'trainingdata' if conbine else file_name
        save_dataset(ELEMENT_LIST, file_name)

if __name__ == "__main__":
    
    hparams = HParams().hparams
    
#    generate_dataset(hparams, CORPUS_DATA_QA_DIR, dialogue_type = 'QA', types = 'txt', conbine = False)
#    generate_dataset(hparams, CORPUS_DATA_MTEM_DIR, dialogue_type = 'MTEM', types = 'json', conbine = False)
    
    # Load tfrecord
    
#    loader = DataLoader(hparams, training=True, mode = 'ddpg')
#
#    batch_input = loader.get_training_batch(loader.train_dataset)
#    initializer = tf.random_uniform_initializer(-0.1, 0.1, seed= 0.1)
#    tf.get_variable_scope().set_initializer(initializer)
#        
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        sess.run(tf.tables_initializer())
#        
#        sess.run(batch_input.initializer)
#        output = sess.run([batch_input.source, batch_input.target, batch_input.addition])
    