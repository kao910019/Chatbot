# -*- coding: utf-8 -*-
import os
import time

#=====================================================================
SYSTEM_ROOT = os.path.abspath(os.path.dirname(__file__))
SYSTEM_FILE = os.path.join(SYSTEM_ROOT, "main.py")
SYSTEM_STRING = []
SYSTEM_DEBUG = [False]
SYSTEM_CLOSE = [False]
SYSTEM_TIME = time.asctime(time.localtime(time.time()))
SYSTEM_TICK = lambda : time.time()

def Debug_message(message):
    if SYSTEM_DEBUG[0]:
        SYSTEM_STRING.append(message)
    print(message)

#=====================================================================
# Bert
""" 
    cased_L-12_H-768_A-12
    uncased_L-12_H-768_A-12 
    wwm_cased_L-24_H-1024_A-16
    wwm_uncased_L-24_H-1024_A-16
"""
BERT_DIR = os.path.join(SYSTEM_ROOT, "Bert")
BERT_PARAMS_DIR = os.path.join(BERT_DIR, "uncased_L-12_H-768_A-12")
BERT_PARAMS_FILE = os.path.join(BERT_PARAMS_DIR, "bert_model.ckpt")
BERT_CONFIG_FILE = os.path.join(BERT_PARAMS_DIR, "bert_config.json")

VOCAB_FILE = os.path.join(BERT_PARAMS_DIR, 'vocab.txt')

#=====================================================================
# PATH

CORPUS_DIR = os.path.join(SYSTEM_ROOT, 'Corpus')
EMOTION_FILE = os.path.join(CORPUS_DIR, 'emotion_vocab.txt')
CORPUS_DATA_DIR = os.path.join(CORPUS_DIR, 'Data')
CORPUS_DATA_MTEM_DIR = os.path.join(CORPUS_DATA_DIR, 'MTEM')
CORPUS_DATA_QA_DIR = os.path.join(CORPUS_DATA_DIR, 'QA')

RECORD_DIR = os.path.join(CORPUS_DIR, 'tfrecord')

RESULT_DIR = os.path.join(SYSTEM_ROOT, 'Result')
RESULT_FILE = os.path.join(RESULT_DIR, 'save')
METADATA_FILE = os.path.join(RESULT_DIR, 'metadata.tsv')

INFER_LOG_DIR = os.path.join(RESULT_DIR, "infer_log")
TRAIN_LOG_DIR = os.path.join(RESULT_DIR, "train_log_")
TEST_LOG_DIR = os.path.join(RESULT_DIR, "test_log_")

TEMP_DIR = os.path.join(SYSTEM_ROOT, 'tmp')

SYSTEM_ROOT, BERT_PARAMS_FILE, BERT_CONFIG_FILE, VOCAB_FILE

#=====================================================================
# NLP

SUBJECT_LIST = ['nsubj','nsubjpass','csubj','csubjpass']
OBJECT_LIST = ['dobj','pobj']

PERSON_PRONOUN = ['he','she','it','they','you']
NOUN_LIST = ['PROPN','NOUN']

PRON_DICT = {"PERSON":['he', 'she', 'they', 'you'],
             "NP":['it', 'they', 'he', 'she', 'you']}

STOPWORDS_LIST = ['the', '..', '--']

VOCAB_DICT = {'[PAD]':'pad',
              # triples
              '[unused1]':'subject',
              '[unused2]':'subject_type',
              '[unused3]':'object',
              '[unused4]':'object_type',
              '[unused5]':'relation',
              # functions
              '[unused6]':'none',
              '[unused7]':'greeting',
              '[unused8]':'question',
              '[unused9]':'affirmative',
              '[unused10]':'negative',
              '[unused11]':'random'}

#=====================================================================
# Neural Network
NN_NAME = "My_NN"
#Database
DATABASE_TYPES = ['PERSON','THING','TIME','LOCATION','ITEM',
                  'REMEMBER','ORGANIZATION','NEURAL_NETWORK',
                  'EMBEDDING']
#QA file list
SPLIT_LIST = ["'m","'s","'t","'ll","'re","~","$",",",".","!","?"]

TRAIN_MODE_LIST = ['pretrain','classifier','adversarial']
                
QA_DATA_FOLDER = os.path.join(CORPUS_DIR, 'Data')
QA_FOLDER = ['Augment0','Augment1','Augment2']
TRAIN_FOLDER = os.path.join(QA_DATA_FOLDER, 'Traindata')
TRAIN_FILE = os.path.join(TRAIN_FOLDER, 'train_data.txt')
TEST_FILE = os.path.join(TRAIN_FOLDER, 'test_data.txt')

RECORD_FILE_NAME_LIST = ['original','emotionpush','friends']

FUNCTION_FOLDER = []

DULL_RESPONSE = ["oh , I don't know what you're talking about.",
                 "I don't know.", "You don't know.",
                 "You know what I mean.",
                 "oh , i ' s a right",
                 "You know what I'm saying.",
                 "You don't know anything.",
                 "I'm not sure what I'm not sure.",
                 "oh , I'm not sure what you mean.",
                 "I'm sorry."]

#Emotion
EMOTION_TYPES = ['Like','Happy','Angry','Sad','Fear']
EMOTION_LENGTH = len(EMOTION_TYPES)
#=====================================================================
Standford_Parser_Dir = os.path.join(SYSTEM_ROOT, "StanfordNLP", "models" )
Standford_Models_Dir = os.path.join(SYSTEM_ROOT, "StanfordNLP", "models")
Standford_Class_Dir = os.path.join(SYSTEM_ROOT, "StanfordNLP", "jars")
Java_Home_Dir = "C:/Program Files/Java/jre1.8.0_191"

#=====================================================================
#Discord Admin ID:
ADMIN_ID = 311784266788896793,246574877165748224,289666561000734721
#=====================================================================

BOT_NAME = "Lamibot"
COMMAND_PREFIX = ["!","?"]

def Lamibot_register(): 
# ====================================================================
# TOKEN = NTAxNzMyNzQyOTI5NzExMTM1.DqdsUA.6CqXs3qF4hHiqr9_OYztt3N4N9s
# ID = 501732742929711135
# PERMISSIONS NUMBER = 67648
# URL = https://discordapp.com/oauth2/authorize?client_id=501732742929711135&scope=bot&permissions=67648
# ====================================================================
    TOKEN = "NTAxNzMyNzQyOTI5NzExMTM1.DqdsUA.6CqXs3qF4hHiqr9_OYztt3N4N9s"
    ID = "501732742929711135"
    PERMISSIONS_NUMBER = "8"
    register = (TOKEN,ID,PERMISSIONS_NUMBER)
    return register
#=====================================================================

#=====================================================================
CREDENTIAL_DIR = os.path.join(SYSTEM_ROOT, '__credentials__')
UPLOAD_DIR = os.path.join(SYSTEM_ROOT, 'downloads')

WOLOLO_DL_ID = "13Z4Cfjj4B5Dey4z_uvlP3cFhnweRoxCn"
Lamibot_ID = '0AG3oNK9IXyheUk9PVA'
WOLOLO_ID = '0ACp9u2Nxb6WnUk9PVA'

def GoogleDrive_register():
    SCOPES = 'https://www.googleapis.com/auth/drive'
    CLIENT_SECRET_FILE = 'GoogleDrive/client_secret.json'
    APPLICATION_NAME = 'Lamibot with Drive API'
    register = (SCOPES,CLIENT_SECRET_FILE,APPLICATION_NAME)
    return register
#=====================================================================