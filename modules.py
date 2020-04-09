# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''
import os
import codecs
import numpy as np
import tensorflow as tf

import io
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.python import pywrap_tensorflow

def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def scaled_dot_product_attention(Q, K, V, key_masks, 
                                 share_softmax = None, share_attention = None, 
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        softmax = share_softmax if share_softmax!=None else tf.nn.softmax(outputs)
        
#        attention = tf.transpose(outputs, [0, 2, 1])
        # attention_summary = tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
        

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(softmax, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        attention = share_attention if share_attention!=None else tf.matmul(outputs, V)  # (N, T_q, d_v)
    return softmax, attention

def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.maximum(tf.shape(key_masks)[0], 1), 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
        
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

def multihead_attention(queries, keys, values, key_masks, 
                        cache=None, memory_cache=False, while_loop=None, share_softmax = None, share_attention = None,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention",
                        trainable=True):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    def compute(inputs, d_model, num_heads, name, trainable):
        # Linear projections
        output = tf.layers.dense(inputs, d_model, name=name, trainable=trainable) # (N, T_q, d_model)
        # Split and concat
        output = tf.concat(tf.split(output, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        return output
    
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = compute(queries, d_model, num_heads, name='query', trainable=trainable)
        
        if cache is None:
            K = compute(keys, d_model, num_heads, name='key', trainable=trainable)
            V = compute(values, d_model, num_heads, name='value', trainable=trainable)
        else:
            if memory_cache:
                if while_loop is None:
                    K = compute(keys, d_model, num_heads, name='key', trainable=trainable)
                    V = compute(values, d_model, num_heads, name='value', trainable=trainable)
                else:
                    K = cache[0]
                    V = cache[1]
            else:
                K = compute(keys[:,-1:,:], d_model, num_heads, name='key', trainable=trainable)
                V = compute(values[:,-1:,:], d_model, num_heads, name='value', trainable=trainable)
                K = tf.concat([cache[0], K], axis=1)
                V = tf.concat([cache[1], V], axis=1)
                
        # Attention
#        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)
        softmax, attention = scaled_dot_product_attention(Q, K, V, key_masks, share_softmax, share_attention, causality, dropout_rate, training)
        
        # Restore shape
        outputs = tf.concat(tf.split(attention, num_heads, axis=0), axis=2) # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
        
#    return outputs
    return outputs, tf.convert_to_tensor([K, V]), softmax, attention

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def ff(inputs, num_units, scope="positionwise_feedforward", trainable=True):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=gelu, trainable=trainable)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1], trainable=trainable)

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def build_gradient(hparams, loss, params, step=None, clip_grad=True, warmup_steps=False, learning_rate=None):
    learning_rate = hparams.init_learning_rate if learning_rate == None else learning_rate
    learning_rate = noam_scheme(learning_rate, step, hparams.warmup_steps) if warmup_steps else learning_rate
    gradients = tf.gradients(loss, params)
    grad, _ = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm) if clip_grad else gradients
    opt = tf.train.AdamOptimizer(learning_rate)
    update = opt.apply_gradients(zip(grad, params), global_step = step)
    return update
"""
Tensorflow实现何凯明的Focal Loss, 该损失函数主要用于解决分类问题中的类别不平衡
focal_loss_sigmoid: 二分类loss
focal_loss_softmax: 多分类loss
Reference Paper : Focal Loss for Dense Object Detection
"""

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)**gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits, axis=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=-1)
    return L

#=================================================================================================
#       Model
#=================================================================================================
def slice_multiterm_function(index, source, target):
    # [batch_size, terms, id]
    output_source = tuple(i[:,index] for i in source)
    output_target = tuple(i[:,index] for i in target)
    return output_source, output_target
    
def build_rnn_cell(hparams, dropout_rate, trainable=True):
    """Create multi-layer RNN cell."""
    cell_list = []
    for i in range(hparams.num_rnn_layers):
        #Create a single RNN cell
        if hparams.rnn_cell_type == 'LSTM':
            single_cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_dimensions, state_is_tuple=True, trainable=trainable)
        else:
            single_cell = tf.contrib.rnn.GRUCell(hparams.num_dimensions, trainable=trainable)
        if hparams.dropout_rate != 0.0:
            single_cell = tf.contrib.rnn.DropoutWrapper(cell = single_cell, input_keep_prob = (1.0 - dropout_rate))
        cell_list.append(single_cell)
    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)

def build_rnn_encoder(hparams, rnn_input, initial_state=None, embedding=None, passing_flag=None, length=None, cell=None, input_layer=False, bidirection=False, dropout_rate=0.0, time_major=False, trainable=True, scope='rnn_encoder'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        rnn_input = tf.nn.embedding_lookup(embedding, rnn_input) if embedding != None else rnn_input
        rnn_input = tf.layers.dense(rnn_input, hparams.num_dimensions, trainable=trainable) if input_layer else rnn_input
        cell = build_rnn_cell(hparams, dropout_rate, trainable=trainable) if cell == None else cell
        initial_state = tuple([initial_state[:,index,:] for index in range(hparams.num_rnn_layers)]) if initial_state != None else initial_state
        initial_state = tf.cond(passing_flag, lambda: initial_state, lambda: cell.zero_state(tf.shape(rnn_input)[0], dtype=tf.float32)) if passing_flag != None else initial_state
        rnn_output, rnn_state = tf.nn.dynamic_rnn(cell, rnn_input, initial_state=initial_state, dtype=tf.float32, sequence_length=length, swap_memory=True, time_major=time_major)
        rnn_state = tf.concat([tf.expand_dims(state, axis=1) for state in rnn_state], axis=1) # [batch_size, layer_num, dim]
    return rnn_output, rnn_state, cell

def build_liner_classifier(hparams, inputs, num_class, activation=gelu, trainable=True):
    logits = tf.layers.dense(ln(inputs), hparams.num_dimensions, activation=gelu, trainable=trainable)
    logits = tf.layers.dense(logits, num_class, use_bias=False, trainable=trainable)
    return logits

def evaluate_classifier(labels, logits, num_class, num_pad=0, name_list=None, name="classifier"):
    with tf.variable_scope('evaluate_'+name, reuse=tf.AUTO_REUSE):
        precision, recall, f1 = [0 for _ in range(num_class-num_pad)], [0 for _ in range(num_class-num_pad)], [0 for _ in range(num_class-num_pad)]
        confusion_matrix = tf.cast(tf.math.confusion_matrix(labels, tf.argmax(logits, axis=-1, output_type=tf.int32), num_classes=num_class, name=name+"_confusion_matrix"), tf.float32)[num_pad:,num_pad:]
        name_list = range(num_class) if name_list is None else name_list
        for i in range(num_class-num_pad):
            precision[i] = tf.divide(confusion_matrix[i,i], tf.reduce_sum(confusion_matrix[:,i]), name=name+"_{}_precision".format(name_list[i+num_pad]))
            recall[i] = tf.divide(confusion_matrix[i,i], tf.reduce_sum(confusion_matrix[i,:]), name=name+"_{}_recall".format(name_list[i+num_pad]))
            f1[i]= tf.divide(2.0 * recall[i] * precision[i], (recall[i] + precision[i]), name=name+"_{}_f1".format(name_list[i+num_pad]))
        accuracy = tf.divide(tf.reduce_sum(tf.eye(num_class-num_pad) * confusion_matrix), tf.reduce_sum(confusion_matrix), name=name+"_accuracy")
        evaluate = check_nan([accuracy]+precision+recall+f1)
    return evaluate
            

def build_discriminator(hparams, embedding, source_id, source_length, target_id, target_length, dropout_rate):
    batch_size, max_time = tf.shape(target_id)[0], tf.shape(target_id)[1]
    with tf.variable_scope('Discriminator_Reward', reuse=tf.AUTO_REUSE):
        #Build Question Encoder
        with tf.variable_scope('Encoder_source'):
            ques_output, ques_state, _ = build_rnn_encoder( 
                hparams, source_id, embedding=embedding, length=source_length, dropout_rate=dropout_rate)
        #Build Response Encoder
        with tf.variable_scope('Encoder_target'):
            resp_output, resp_state, _ = build_rnn_encoder(
                    hparams, target_id, initial_state=ques_state, embedding=embedding, length=target_length, dropout_rate=dropout_rate)
        logits = build_liner_classifier(hparams, resp_output, 2)
        labels = tf.concat([tf.fill([batch_size/2, max_time], 0), tf.fill([batch_size/2, max_time], 1)], axis=0)
        
        # target_length = tf.where(tf.less_equal(target_length, 1), tf.zeros_like(target_length), target_length)
        target = tf.sequence_mask(target_length, max_time, dtype=tf.float32)
    return labels, target, logits

# def regroup_sentence(source_id, target_id, max_batch_size):
#     """
#     Combine and regroup source_id and target_id from N-toN form to One-to-One form,
#     outputs term_length and max_length will changed.
    
#     Parameters
#     ----------
#     source_id : Tensor. shape = [term_length, max_length]
#     target_id : Tensor. shape = [term_length, max_length]
#     max_batch_size : Int32 or Int64 (can't input a tensor)

#     Returns
#     -------
#     sentence_regroup : Tensor. shape = [term_length, max_length]
#     """
#     with tf.name_scope("regroup_sentence"):
#         sentence_ids = tf.concat([source_id, target_id], axis= 1)
#         batch_size, max_length = tf.shape(sentence_ids)[0], tf.shape(sentence_ids)[1]
        
#         source_indice = tf.where(tf.not_equal(source_id[:,0], 0))
#         target_indice = tf.where(tf.not_equal(target_id[:,0], 0))
#         source_indice = tf.scatter_nd(source_indice, tf.ones_like(source_indice, tf.int32), [batch_size, 1])
#         target_indice = tf.scatter_nd(target_indice, tf.ones_like(target_indice, tf.int32), [batch_size, 1])
#         target_indice = tf.roll(target_indice, shift=1, axis=0)
        
#         part_indice = tf.roll(tf.reshape(source_indice * target_indice, [-1]), shift=-1, axis=0)
#         part_indice = tf.scan(lambda a,x : a+x, part_indice, reverse = True)
#         part_indice = (tf.reduce_max(part_indice) - part_indice)
    
#         term_count = tf.reduce_max(tf.unique_with_counts(part_indice)[2])
    
#         sentence_part = tf.dynamic_partition(sentence_ids, part_indice, max_batch_size)
#         sentence_regroup = tf.concat([tf.cond(tf.not_equal(tf.size(part), 0), 
#                             lambda: tf.concat([tf.reshape(part, [1, -1]), tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)], axis=1), 
#                             lambda: tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)) for part in sentence_part], 0)
        
#         indice = tf.reshape(tf.where(tf.not_equal(sentence_regroup[:,0], 0)), [-1])
#         sentence_regroup = tf.gather(sentence_regroup, indice)
#     return sentence_regroup

def combine_bert_input(input_id_eval, input_id_target, input_length_eval, input_length_target, flag):
    # def pad_and_concat(input_id_eval, input_id_target):
    #     input_id_eval, input_id_target, _ = pad_complement_each_other(input_id_eval, input_id_target, 0, axis=1)
    #     return tf.concat([input_id_eval, input_id_target], axis=0)
    # output_id = tf.cond(tf.equal(flag, False), lambda: input_id_eval, lambda: pad_and_concat(input_id_eval, input_id_target))
    output_id = tf.cond(tf.equal(flag, False), lambda: input_id_eval, lambda: tf.concat([input_id_eval, input_id_target], axis=0))
    output_length = tf.cond(tf.equal(flag, False), lambda: input_length_eval, lambda: tf.concat([input_length_eval, input_length_target], axis=0))
    return output_id, output_length

def split_bert_output(bert, batch_size, flag):
    vector_eval = tf.cond(tf.equal(flag, False), lambda: bert.finetuning_vector, lambda: bert.finetuning_vector[:batch_size])
    context_eval = tf.cond(tf.equal(flag, False), lambda: bert.context_vector, lambda: bert.context_vector[:batch_size])
    vector_target = tf.cond(tf.equal(flag, False), lambda: bert.finetuning_vector, lambda: bert.finetuning_vector[batch_size:])
    context_target = tf.cond(tf.equal(flag, False), lambda: bert.context_vector, lambda: bert.context_vector[batch_size:])
    return vector_eval, context_eval, vector_target, context_target
    
def resort_zero(inputs, max_batch_size):
    """  
    Inputs 2D tensor that can resort zero in tensor to the end.
    
    Parameters
    ----------
    inputs : Tensor. shape = [batch_size, max_length]
    max_batch_size : Int32 or Int64 (can't input a tensor)
    
    Returns
    -------
    outputs : Tensor. shape = [batch_size, max_length]
    """
    with tf.name_scope("resort_zero"):
        batch_size, max_length = tf.shape(inputs)[0], tf.shape(inputs)[1]
        odd_indices = tf.matmul(tf.reshape(tf.range(1, batch_size*2, 2), [-1,1]), tf.fill([1,max_length], 1))
        even_indices = tf.matmul(tf.reshape(tf.range(0, batch_size*2, 2), [-1,1]), tf.fill([1,max_length], 1))
        partitions = tf.where(tf.not_equal(inputs, 0), even_indices, odd_indices)
        sequence_parts = tf.dynamic_partition(inputs, partitions, max_batch_size*4)
        outputs = tf.reshape(tf.concat(sequence_parts, axis=0), tf.shape(inputs))
    return outputs, partitions
    
def pad_complement_each_other(inputs1, inputs2, pad=None, axis=None):
    max_axis_shape = tf.reduce_max([tf.shape(inputs1)[axis], tf.shape(inputs2)[axis]])
    inputs1 = tf.concat([inputs1, tf.fill([max_axis_shape - tf.shape(inputs1)[i] if i==axis else tf.shape(inputs1)[i] for i in range(tf.shape(inputs1).shape[0])], pad)], axis=axis)
    inputs2 = tf.concat([inputs2, tf.fill([max_axis_shape - tf.shape(inputs2)[i] if i==axis else tf.shape(inputs2)[i] for i in range(tf.shape(inputs2).shape[0])], pad)], axis=axis)
    return inputs1, inputs2, max_axis_shape

def summary_sentence_regroup(source_id, target_id, pad_id, source_tag='Source', target_tag='Target'):
    batch_size = tf.shape(source_id)[0]
    
    source_id, target_id, max_time = pad_complement_each_other(source_id, target_id, pad_id, axis=1)
    sentence_ids = tf.reshape(tf.concat([source_id, target_id], axis= 1), [-1, max_time])
    zero_indice = tf.where(tf.not_equal(sentence_ids[:, 0], 0))
    
    tag_labels = tf.reshape(tf.concat([tf.fill([batch_size, 1], 0), tf.fill([batch_size, 1], 1)], axis=1), [-1, 1])
    
    sentence_ids = tf.gather_nd(sentence_ids, zero_indice)
    tag_labels = tf.gather_nd(tag_labels, zero_indice)
    
    label_batch_size = tf.shape(tag_labels)[0]
    tag_labels = tf.where(tf.equal(tag_labels, 0), tf.fill([label_batch_size, 1], source_tag), tf.fill([label_batch_size, 1], target_tag))
    return tag_labels, sentence_ids

def regroup_sentence_one2one_form(hparams, source_id, target_id):
    # sentence_regroup
    tag, sentence_id = summary_sentence_regroup(source_id, target_id, 0, source_tag=False, target_tag=True)
    max_length = tf.shape(sentence_id)[1]
    # sentence_slice
    front_slice_indice = tf.where(tf.equal(tag, False))[0,0]
    back_slice_indice = tf.where(tf.equal(tag, True))[-1,0] + 1
    sentence_id = sentence_id[front_slice_indice:back_slice_indice]
    tag = tag[front_slice_indice:back_slice_indice]
    # part indice
    tag = tf.cast(tf.logical_xor(tag, tf.roll(tag, shift=-1, axis=0)), tf.int32)
    part_indice = tf.scan(lambda a,x : a+x, tag, reverse = True)
    part_indice = tf.reshape(tf.reduce_max(part_indice) - part_indice, [-1])
    term_count = tf.reduce_max(tf.unique_with_counts(part_indice)[2])
    # regroup part sentence
    sentence_part = tf.dynamic_partition(sentence_id, part_indice, hparams.batch_size)
    sentence_regroup = tf.concat([tf.cond(tf.not_equal(tf.size(part), 0), 
                        lambda: tf.concat([tf.reshape(part, [1, -1]), tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)], axis=1), 
                        lambda: tf.zeros([1, (term_count - tf.shape(part)[0])*max_length], tf.int32)) for part in sentence_part], 0)
    # sort zero to behind
    sentence_regroup, _ = resort_zero(sentence_regroup, hparams.batch_size)
    # remove pad sentence
    indice = tf.reshape(tf.where(tf.not_equal(sentence_regroup[:,0], 0)), [-1])
    sentence_regroup = tf.gather(sentence_regroup, indice)
    # split two part
    max_length = tf.shape(sentence_regroup)[1]
    sentence_regroup = tf.reshape(sentence_regroup, [-1, max_length*2])
    source_id = sentence_regroup[:,:max_length]
    target_id = sentence_regroup[:,max_length:]
    return source_id, target_id

def one2one_form_summary(hparams, reverse_table, source_id, target_id, predict_id, pad, target=False):
    # For predict sentence
    # indice = tf.random_uniform((), 0, tf.cast(self.batch_size/self.term_length_placeholder, tf.int32), tf.int32) * self.term_length_placeholder
    source_id = source_id[:,1:]
    predict_tag_labels, predict_sentence_id = summary_sentence_regroup(source_id, predict_id, pad, source_tag='Source', target_tag='Predict')
    predict_sentence = reverse_table.lookup(tf.to_int64(predict_sentence_id))
    predict_sentence = tf.strings.regex_replace(tf.strings.reduce_join(predict_sentence, separator=' ', axis=-1), " ##|\\[PAD\\]", "")
    predict_sentence = tf.concat([predict_tag_labels, tf.reshape(predict_sentence, [-1,1])], axis=1)
    predict_sentence = tf.strings.reduce_join(predict_sentence, separator=' : ', axis=-1)
    
    if target:
        target_tag_labels, target_sentence_id = summary_sentence_regroup(source_id, target_id, pad, source_tag='Source', target_tag='Target')
        target_sentence = reverse_table.lookup(tf.to_int64(target_sentence_id))
        target_sentence = tf.strings.regex_replace(tf.strings.reduce_join(target_sentence, separator=' ', axis=-1), " ##|\\[PAD\\]", "")
        target_sentence = tf.concat([target_tag_labels, tf.reshape(target_sentence, [-1,1])], axis=1)
        target_sentence = tf.strings.reduce_join(target_sentence, separator=' : ', axis=-1)
        predict_sentence, target_sentence, _ = pad_complement_each_other(predict_sentence, target_sentence, pad="", axis=0)
        output_string = tf.transpose(tf.convert_to_tensor([target_sentence, predict_sentence]))
    else:
        output_string = tf.transpose(tf.convert_to_tensor([predict_sentence]))
        
    evaluation_summary = tf.summary.merge([tf.summary.text("Dialogue summary", output_string)])
    return evaluation_summary

def rebuild_mtem(mtem_inputs, term_inputs, axis=0):
    if mtem_inputs == tuple():
        return tuple(np.expand_dims(ti, axis=axis) for ti in term_inputs)
    else:
        pad_bs = mtem_inputs[0].shape[0] - term_inputs[0].shape[0]
        if pad_bs != 0:
            pad_inputs = tuple(np.zeros_like(mi[:,0])[:pad_bs] for mi in mtem_inputs)
            term_inputs = tuple(np.concatenate([ti, pi], axis=0)for ti, pi in zip(term_inputs, pad_inputs))
        return tuple(np.concatenate([mi, np.expand_dims(ti, axis=axis)], axis=axis) for mi, ti in zip(mtem_inputs, term_inputs))

def add_cls(hparams, inputs):
    new_inputs = tuple()
    new_inputs += (inputs[0],)
    new_inputs += (np.concatenate([np.expand_dims((hparams.cls_id * np.not_equal(inputs[2], 0).astype(np.int32)), 1), inputs[1]], axis=1)[:,:hparams.max_length],)
    new_inputs += (np.minimum(np.where(inputs[2] != 0, inputs[2]+1, inputs[2]), hparams.max_length),)
    return new_inputs

def padding_action(hparams, inputs, ac_flag=True):
    new_inputs = tuple()
    if ac_flag:
        inputs = add_cls(hparams, inputs)
    for inp in inputs:
        if len(inp.shape) > 1:
            zeros = np.zeros(list(np.maximum(hparams.max_length - i, 0) if index == 1 else i for index, i in enumerate(inp.shape)), dtype=inp.dtype)
            inp = np.concatenate((inp, zeros), axis=1)
        new_inputs += (inp,)
    return new_inputs
    
def filter_complete_dialogue(flag1, flag2, actions, states, last_indice=None):
    indice = np.where((flag1+flag2)!=0)
    new_actions = tuple()
    for action in actions:
        new_actions += (tuple(a[indice] for a in action),)
    new_states = tuple()
    for state in states:
        if last_indice!=None:
            new_states += (state[np.where((flag1+flag2)[last_indice]!=0)],)
        else:
            new_states += (state[indice],)
    return new_actions, new_states, indice
        
        

def get_memory_replay(actions, states, rewards, memory, batch_size, max_memory_size=1000):
    batch_actions, batch_states, batch_rewards = tuple(), tuple(), tuple()
    new_memory = {'action':tuple(),'state':tuple(),'rewards':tuple()}
    # Rewards
    first = False
    if memory['rewards'] == tuple():
        first, memory['rewards'] = True, rewards
    max_size = len(memory['rewards'][0])
    batch_size = max_size if batch_size > max_size else batch_size
    indices = np.random.choice(max_size, size=batch_size)
    for reward, mem_reward in zip(rewards, memory['rewards']):
        new_reward = np.concatenate((reward, mem_reward), axis=0)[:max_memory_size] if not first else mem_reward
        new_memory['rewards'] += (new_reward,)
        batch_rewards += (new_reward[indices],)
    # Action
    first = False
    if memory['action'] == tuple():
        first, memory['action'] = True, actions
    for action, mem_action in zip(actions, memory['action']):
        new_action = tuple(np.concatenate((act, mact), axis=0)[:max_memory_size] for act, mact in zip(action, mem_action)) if not first else mem_action
        new_memory['action'] += (new_action,)
        batch_actions += (tuple(act[indices] for act in new_action),)
    # State
    first = False
    if memory['state'] == tuple():
        first, memory['state'] = True, states
    for state, mem_state in zip(states, memory['state']):
        new_state = np.concatenate((state, mem_state), axis=0)[:max_memory_size] if not first else mem_state
        new_memory['state'] += (new_state,)
        batch_states += (new_state[indices],)
    return batch_actions, batch_states, batch_rewards, new_memory

#=================================================================================================
#       Initializer
#=================================================================================================
def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.contrib.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


#=================================================================================================
#       Vocab save & load
#=================================================================================================
def save_vocab(file_name, vocab_list):
    try:
        os.remove(file_name)
    except:
        pass
    with open(file_name, 'a', encoding = 'utf8') as f:
        for vocab in vocab_list:
            f.write("{}\n".format(vocab))
    print("Save {} file.".format(file_name))
    
def check_vocab(file_path):
    if tf.gfile.Exists(file_path):
        vocab_list = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(file_path, "rb")) as f:
            for word in f:
                vocab_list.append(word.strip())
    else:
        raise ValueError("The vocab_file does not exist.")
    return len(vocab_list), vocab_list

def rebuild_vocab(corpus_file):
    vocab_list = []
    with open(corpus_file, 'r', encoding = 'utf8') as f:
        for line in tqdm(f):
            l = line.strip()
            if not l:
                continue
            tokens = l.strip().split(' ')
            for token in tokens:
                if len(token) and token != ' ':
                    t = token.lower()
                    if t not in vocab_list:
                        vocab_list.append(t)
    save_vocab(corpus_file, vocab_list)
    return vocab_list
#=================================================================================================
#       Data save & load
#=================================================================================================
def get_tensors_in_checkpoint_file(file_name, all_tensors=True, tensor_name=None):
    varlist, var_value = [], []
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = []
    for i, tensor_name in enumerate(loaded_tensors[0]):
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
            full_var_list.append(tensor_aux)
        except:
            print('* Not found: '+tensor_name)
    return full_var_list

def variable_loader(session, result_dir, var_list = tf.global_variables(), max_to_keep=5):
    ckpt = tf.train.get_checkpoint_state(result_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("# Find checkpoint file:", ckpt.model_checkpoint_path)
        restored_vars  = get_tensors_in_checkpoint_file(file_name = ckpt.model_checkpoint_path)
        tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
        saver = tf.train.Saver(tensors_to_load, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=1.0)
        print("# Restoring model weights ...")
        saver.restore(session, ckpt.model_checkpoint_path)
        return saver, True
    saver = tf.train.Saver(var_list, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=1.0)
    return saver, False
#=================================================================================================
#       Summary log
#=================================================================================================
def reward_plot(rewards, rewards_name, term_length, scale=None, title=None):
    x, total_reward = np.arange(1, term_length+1), 0
    plot_list = []
    for index, (reward, name) in enumerate(zip(rewards, rewards_name)):
        avg_reward = reward
        if scale != None:
            avg_reward *= scale[index]
        total_reward += avg_reward
        plot_list.append((x, avg_reward, '-', name))
    plot_list.append((x, total_reward, '-', "total reward"))
    return generate_plot(plot_list, title=title, xlabel='Term', ylabel='Reward')

def generate_plot(inputs, title=None, xlabel=None, ylabel=None):
    # Inputs a list contain tuple(x, y, line_type, label)
    plt.figure()
    for x, y, line, label in inputs:
        plt.plot(x, y, line, label=label)
    if title != None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    plt.legend(loc='upper left', framealpha=0.5, prop={'size': 'xx-small'})
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close('all')
    buffer.seek(0)
    return buffer.getvalue()

def image_scalar(name, byte_string_tensor):
    image = tf.image.decode_png(byte_string_tensor, channels=4)
    image = tf.expand_dims(image, 0)
    return tf.summary.image(name, image)

def time_string(time):
    time = round(time, 2)
    return "{:02d}:{:02d}:{:02d}".format(int(time/3600), int(time%3600/60), int(time%60))
    
def check_nan(variable_list, default_value = None):
    new_variable_list = []
    for variable in variable_list:
        default_value = tf.zeros_like(variable) if default_value == None else default_value * tf.ones_like(variable)
        new_variable_list.append(tf.where(tf.is_nan(variable), default_value, variable, name = variable.op.name.split('/')[-1]))
    return new_variable_list

def get_sentence_length(ids, batch_size=None, end_id=0, max_length=50, name=None):
    batch_size = batch_size if batch_size != None else tf.shape(ids)[0]
    length = tf.where(tf.less(ids-end_id, 0), tf.ones_like(ids), ids-end_id)
    length = tf.minimum(tf.cast(tf.argmin(tf.concat([length, tf.fill([batch_size, 1], 0)], axis=-1), -1), tf.int32) + 1, max_length, name=name)
    return length

def avg_variable_list(avg_scale, variable_list):
    return [tf.identity((variable * avg_scale), name=variable.op.name.split('/')[-1]) for variable in variable_list]

def log_variable(variable_list, total_batch_size, scope = 'Outputs_summary'):
    with tf.variable_scope(scope):
        epoch_summary_dict = {}
        epoch_summary_list = []
        log_variable_list = []
        for variable in variable_list:
            name = variable.op.name.split('/')[-1]
            log_var = tf.Variable(0.0, name=name, trainable=False)
            epoch_summary_dict.update({name: 0.0})
            epoch_summary_list.append(tf.summary.scalar(name, log_var / tf.cast(tf.maximum(total_batch_size, 1), tf.float32)))
            log_variable_list.append(log_var)
        epoch_summary_op = tf.summary.merge(epoch_summary_list)
    return epoch_summary_op, epoch_summary_dict, log_variable_list

def summary_calculate(summary_dict, log_variable_list, result_list):
    for variable, result in zip(log_variable_list, result_list):
        name = variable.op.name.split('/')[-1]
        summary_dict.update({name: summary_dict[name] + result})
    return summary_dict