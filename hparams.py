# -*- coding: utf-8 -*-
import tensorflow as tf

# Server params
# Below command to search whitch port isn't occupy.
# netstat -tulpn | grep LISTEN
flags = tf.app.flags
#flags.DEFINE_string("ps_hosts", "192.168.221.1:2222", "param server hosts")
#flags.DEFINE_string("wk_hosts", "192.168.221.1:2223", "worker server hosts")
flags.DEFINE_string("ps_hosts", "localhost:2222", "param server hosts")
flags.DEFINE_string("wk_hosts", "localhost:2223", "worker server hosts")
flags.DEFINE_string("job_name", None, "'ps' or 'worker'")
flags.DEFINE_integer("task_index", 0, "index of task within the job")
flags.DEFINE_boolean("trace_timeline", False, "to trace timeline when running tensorflow")
flags.DEFINE_boolean("use_gpu", False, "")

# Initialisation params
flags.DEFINE_string("train_mode", 'p', "choose training mode")
flags.DEFINE_string("init_op", "uniform", "init option")
flags.DEFINE_float("init_weight", 0.1, "init weight")
flags.DEFINE_float("random_seed", 0.1, "random seed")
flags.DEFINE_float("max_gradient_norm", 5.0, "max gradient norm")

# Learning params
flags.DEFINE_string("output_activation", "", "tanh / leaky_relu / ")
flags.DEFINE_boolean("test_dataset", True, "use test dataset")
flags.DEFINE_boolean("label_smooth", True, "")
flags.DEFINE_boolean("discount_reward", True, "")
flags.DEFINE_boolean("normalize_reward", True, "")
flags.DEFINE_boolean("emotion_reward", True, "")
flags.DEFINE_boolean("response_reward", True, "")
flags.DEFINE_boolean("split_train_gd", False, "")
flags.DEFINE_boolean("use_rnn_encoder", False, "")
flags.DEFINE_boolean("dynamic_discriminator_learning_rate", True, "")
flags.DEFINE_boolean("train_encoder_by_classifier", False, "")
flags.DEFINE_integer("num_steps", 0, "")
flags.DEFINE_integer("num_epochs", 19, "num epoch you want to train")
flags.DEFINE_integer("max_to_keep", 4, "max save file to keep")
flags.DEFINE_integer("batch_size", 32, "training some data in the same time")
flags.DEFINE_integer("batch_size_infer", 1, "infer a sentence")
flags.DEFINE_integer("warmup_steps", 2000, "warmup step for increase training speed")
flags.DEFINE_integer("max_length", 30, "max sentence length")
flags.DEFINE_integer("max_term_length", 5, "max dialogue length")
flags.DEFINE_float("init_learning_rate", 8e-5, "learning rate for normal")
flags.DEFINE_float("critic_learning_rate",8e-4, "learning rate for critic")
flags.DEFINE_float("dropout_rate", 0.0, "dropout rate when for pretrain")
flags.DEFINE_float("train_generator_threshold", 0.85, "discriminator accuarcy threshold")
flags.DEFINE_float("train_discriminator_threshold", 0.60, "discriminator accuarcy threshold")
flags.DEFINE_float("discriminator_accuarcy_threshold", 0.80, "discriminator accuarcy threshold")

# ddpg params
flags.DEFINE_boolean("target_network", False, "")
flags.DEFINE_boolean("force_replace", False, "force replace target params from eval")
flags.DEFINE_boolean("memory_replay", True, "")
flags.DEFINE_float("accuarcy_copy_threshold", 0.05, "5%")
flags.DEFINE_float("soft_replace", 0.01, "slowly replace target params")
flags.DEFINE_integer("memory_capacity", 1000, "memory replay size")
flags.DEFINE_float("gamma", 0.9, "reduce q value usually higher than reward")
flags.DEFINE_boolean("wolpertinger_policy", True, "use Wolpertinger Policy to search action")
flags.DEFINE_integer("k_nearest", 3, "number percent you want to search nearest neighbor")

# Network params
flags.DEFINE_integer("share_attention", 2, "")
flags.DEFINE_integer("bert_dimensions", 768, "bert hidden layer dimension")
flags.DEFINE_integer("num_dimensions", 512, "hidden layer dimension")
flags.DEFINE_integer("num_ff_dimensions", 1024, "feed forward dimension")
flags.DEFINE_integer("num_blocks", 8, "transformer bloks number")
flags.DEFINE_integer("num_heads", 8, "transformer multi-head number")
flags.DEFINE_integer("num_rnn_layers", 2, "RNN multi-layer number")
flags.DEFINE_integer("beam_width", 5, "")
flags.DEFINE_string("rnn_cell_type", "GRU", "'LSTM' or 'GRU' cell type for RNN")

# Tokenizer params
flags.DEFINE_string("tokenize_mode", "Full", "'Basic' or 'Full' mode to tokenize sentence")
flags.DEFINE_boolean("do_lower_case", True, "lower case")

# Chatbot params
flags.DEFINE_string("pad_token", "[PAD]", "pad token")
flags.DEFINE_string("unk_token", "[UNK]", "unk token")
flags.DEFINE_string("cls_token", "[CLS]", "cls token")
flags.DEFINE_string("sep_token", "[SEP]", "sep token")
flags.DEFINE_integer("pad_id", 0  , "pad id")
flags.DEFINE_integer("unk_id", 100, "unk id")
flags.DEFINE_integer("cls_id", 101, "cls id")
flags.DEFINE_integer("sep_id", 102, "sep id")
  
hparams = flags.FLAGS
class HParams:
    def __init__(self):
        self.hparams = hparams