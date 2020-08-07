import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import grid_rnn 
import numpy

BEAM_WIDTH = 25
VISUAL_CONCEPT_SIZE = 1000
class ICG_model:
    def __init__(self, features: dict, hyperparameters: dict, is_train: bool):
        batch_size = hyperparameters['batch_size']
        feat_dim = hyperparameters['feat_dim']
        vocab_size = hyperparameters['vocab_size']
        word_emb_size = hyperparameters['word_emb_size']
        model = hyperparameters['model']
        
        # Create the projection layer and the word embedding matrix:
        init = tf.constant_initializer(
            0.01 * numpy.random.uniform(-1, 1, size=(vocab_size, word_emb_size)))
        projection_layer = Dense(vocab_size, use_bias=True,
                                 kernel_initializer=init, activation =None,
                                 name='emb_matrix')
        projection_layer.build((vocab_size, word_emb_size))       
        emb = tf.transpose(projection_layer.trainable_weights[0]) 
        
        # Retrieve information from hyperparameters and a batch of tf records:
        if is_train:
            learning_rate = hyperparameters['learning_rate']
            dropout_keep_rate = hyperparameters['dropout_keep_rate']
            caption_batch, target_batch, data_type_batch, im_id_batch, feat_batch \
                = tf.train.shuffle_batch([features['caption'], features['target'],
                                          features['data_type'], features['im_id'],
                                          features['feat']], batch_size=batch_size,
                                         capacity=2000, min_after_dequeue=200) 
            # The caption_batch and target_batch are sparse matrices.
            # we need to convert the to dense matrices:
            caption_batch_dense = tf.sparse_tensor_to_dense(
                sp_input=caption_batch, default_value=0, validate_indices=True,
                                                      name=None) # B x max_len
            target_batch_dense = tf.sparse_tensor_to_dense(
                sp_input=target_batch, default_value=0, validate_indices=True,
                                                      name=None) # B x max_len 
            # embedding the ids of the captions:
            caption_batch_dense_emb = tf.nn.embedding_lookup(
                emb, caption_batch_dense) # B x max_len x word_emb_size
        else:
            # at the test time we do not have a caption, only image features
            dropout_keep_rate = 1.0
            feat_batch = features['feat']  
        
        # project image features into a space of dimension word_emb_size using
        # a densely-connected layer:
        W_feat = tf.Variable(0.01 * tf.random_normal([feat_dim, word_emb_size]))
        b_feat = tf.Variable(tf.zeros([word_emb_size]))
        feat_proj = tf.tensordot(
            feat_batch, W_feat, [[1], [0]]) + b_feat # B x word_emb_size
        feat_proj = tf.reshape(feat_proj,[-1, 1, word_emb_size]) # B x 1 x word_emb_size 
        feat_proj = tf.nn.dropout(feat_proj, keep_prob=dropout_keep_rate) 
        
        # encoding image features (initializing the decoder LSTM):
        if model == 'icg':    
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=word_emb_size)
        if model == 'icg_deep':
            lstm1 = tf.nn.rnn_cell.LSTMCell(num_units=word_emb_size)
            lstm2 = tf.nn.rnn_cell.LSTMCell(num_units=word_emb_size)           
            lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)         
        _, encoder_final_state = tf.nn.dynamic_rnn(lstm, feat_proj, dtype=tf.float32)    
        if is_train:
            # training the decoder: give the last output of the encoder as an
            # input to the decoder:
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = caption_batch_dense_emb, sequence_length = tf.shape(
                    caption_batch_dense_emb)[1] * tf.ones(
                        [batch_size], dtype=tf.int32),
                        time_major = False)            
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = lstm, helper = training_helper, initial_state = encoder_final_state,
                output_layer = projection_layer)

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder, impute_finished = False, 
                maximum_iterations = 100)
            training_logits = training_decoder_output.rnn_output
            probs = tf.nn.softmax(training_logits)  # B x max_len x vocab_size 
            
            # the sum of the negative log likelihood of the correct word at 
            # each time step is chosen as the loss:
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_batch_dense, logits=training_logits)
            mask = tf.cast(target_batch_dense > 0, dtype=tf.float32)           
            cost_mask = tf.multiply(mask, cost)
            cost_mask_sum = tf.reduce_sum(cost_mask, 1) # B x 1
            cross_entropy = tf.reduce_mean(cost_mask_sum) # 1
            
            # the loss is minimized using RMSprop:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay=0.9)
            train_step = optimizer.minimize(cross_entropy)
            self.__train_step = train_step
            self.__cross_entropy = cross_entropy
            self.__logits = training_logits
            self.__probs = probs
            self.__caption_batch = caption_batch            
        else:
            # at the test time use beam search to generate a caption for an image:
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier= BEAM_WIDTH)   
            start_tokens = tf.zeros([tf.shape(feat_batch)[0]], dtype=tf.int32)
            end_token = tf.constant(1, dtype=tf.int32) 
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                lstm, emb, start_tokens, end_token, tiled_encoder_final_state, 
                BEAM_WIDTH, output_layer=projection_layer, length_penalty_weight=0.0,
                coverage_penalty_weight=0.0, reorder_tensor_arrays=True)
            outputs, _, _= tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=25)
            self.__ids = outputs.predicted_ids 
            self.__scores = outputs[1].scores      
        self.__feat_batch = feat_batch 
        self.__W_emb = emb
        self.__info = encoder_final_state      
       
    # @property decorators:
    @property
    def _train_step(self):
        return self.__train_step

    @property
    def _cross_entropy(self):
        return self.__cross_entropy
    
    @property
    def _logits(self):
        return self.__logits

    @property
    def _W_emb(self):
        return self.__W_emb

    @property
    def _probs(self):
        return self.__probs

    @property
    def _feat_batch(self):
        return self.__feat_batch

    @property
    def _info(self):
        return self.__info

    @property
    def _caption_batch(self):
        return self.__caption_batch
    
    @property
    def _ids(self):
        return self.__ids
    @property
    def _scores(self):
        return self.__scores
###############################################################################
#######################   Attention Model   ###################################
###############################################################################
class ICG_model_att:
    def __init__(self, features: dict, hyperparameters: dict, is_train: bool):
        batch_size = hyperparameters['batch_size']
        grid_size = hyperparameters['grid_size']
        vocab_size = hyperparameters['vocab_size']
        word_emb_size = hyperparameters['word_emb_size']
        
        # Create the projection layer and the word embedding matrix:
        init =  tf.constant_initializer(0.01 * numpy.random.uniform(
            -1, 1, size=(vocab_size, word_emb_size)))
        projection_layer = Dense(vocab_size, use_bias=True, kernel_initializer=init)
        projection_layer.build((vocab_size,word_emb_size))
        emb = tf.transpose(projection_layer.trainable_weights[0])
        
        # Retrieve information from hyperparameters and a batch of tf records:
        if is_train:
            grid_feat_batch, caption_batch, target_batch = tf.train.shuffle_batch(
                [features['grid_feat'], features['caption'], features['target']],
                batch_size=batch_size, capacity=20000, min_after_dequeue=200)
            dropout_keep_rate = hyperparameters['dropout_keep_rate']
            
            # The caption_batch and target_batch are sparse matrices
            # we need to convert the to dense matrices:
            caption_batch_dense = tf.sparse_tensor_to_dense(
                sp_input=caption_batch, default_value=0, validate_indices=True)
            target_batch_dense = tf.sparse_tensor_to_dense(
                sp_input=target_batch, default_value=0, validate_indices=True)
            
            # embedding the ids of the captions:
            caption_batch_dense_emb = tf.nn.embedding_lookup(
                emb, caption_batch_dense) # B x max_len x word_emb_size
        else:
            # at the test time we do not have a caption, only image features
            grid_feat_batch = features['grid_feat']
            dropout_keep_rate = 1.0
        
        grid_emb_size = hyperparameters['grid_emb_size']
        grid_feat_dim = hyperparameters['grid_feat_dim']
                
        self.__grid_feat_batch = grid_feat_batch # B x (grid_size * grid_size * 1024)
        grid_feat_batch = tf.reshape(
            grid_feat_batch, [-1, grid_size * grid_size,
                              grid_feat_dim]) # B x (grid_size * grid_size) x grid_feat_dim
        
        # project image region features into a space of dimension word_emb_size
        # using a densely-connected layer:
        W_grid = tf.Variable(0.01 * tf.random_normal([grid_feat_dim, grid_emb_size]))
        b_grid = tf.Variable(tf.zeros([grid_emb_size]))
        feat_batch_proj = tf.tensordot(grid_feat_batch, W_grid, [[2], [0]]
                                       ) + b_grid  # B x (grid_size * grid_size) x grid_emb_size
        # Adds a Layer Normalization layer:
        feat_batch_proj = tf.contrib.layers.layer_norm(feat_batch_proj)
        feat_batch_proj = tf.nn.dropout(feat_batch_proj, keep_prob=dropout_keep_rate)
        
        init_st = tf.nn.rnn_cell.LSTMStateTuple(
            c=tf.zeros([batch_size,512], dtype=tf.float32), h=tf.zeros([batch_size,512],
                                                                       dtype=tf.float32))       
        attention_depth = word_emb_size            
        if not is_train:
            # at the test time we need to tile the encoder_outputs:
            beam_width = 20
            encoder_outputs = feat_batch_proj
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_width)       
                  
            sequence_length = grid_size * grid_size * tf.ones([batch_size], 
                                                              dtype=tf.int64)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                sequence_length, multiplier=beam_width)
                            
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                init_st, multiplier=beam_width)                           
        else:
            tiled_encoder_outputs = feat_batch_proj
            tiled_sequence_length = None  
            
        # define decoder LSTM with an attention mechanism (Bahdanau):
        lstm = tf.nn.rnn_cell.LSTMCell(word_emb_size)       
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=attention_depth, memory=tiled_encoder_outputs,
            memory_sequence_length=tiled_sequence_length)                    
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            lstm, attention_mechanism, alignment_history=True,
            attention_layer_size=word_emb_size)
        attention_cell = tf.contrib.rnn.DropoutWrapper(
            attention_cell, output_keep_prob=dropout_keep_rate)    
        if is_train:
            # training the decoder:
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = caption_batch_dense_emb, sequence_length = tf.shape(
                    caption_batch_dense_emb)[1] * tf.ones(
                        [batch_size], dtype=tf.int32), time_major = False)
            
            decoder_initial_state2 = attention_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size * 1)
            decoder_initial_state2 = decoder_initial_state2.clone(cell_state=init_st)        
    
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = attention_cell, helper = training_helper, 
                initial_state = decoder_initial_state2, output_layer = projection_layer)

            training_decoder_output, AttentionWrapperState, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder, impute_finished = False, maximum_iterations = 100)
            training_logits = training_decoder_output.rnn_output
            
            # the sum of the negative log likelihood of the correct word at 
            # each time step is chosen as the loss:
            lgt = training_logits
            mask = tf.cast(target_batch_dense > 0, dtype=tf.float32)
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_batch_dense, logits=lgt)
            cost_mask = tf.multiply(mask, cost)
            cost_mask_sum = tf.reduce_sum(cost_mask, 1)
            cross_entropy = tf.reduce_mean(cost_mask_sum)
            
            # the loss is minimized
            learning_rate = hyperparameters['learning_rate']
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)            
            train_step = optimizer.minimize(cross_entropy)
            ind_tensor = tf.range(AttentionWrapperState[4].size())
            
            # get the attention weights for image regions
            att_w = AttentionWrapperState[4].gather(indices = ind_tensor, name=None)
            
            info = [lgt]    
            self.__train_step = train_step
            self.__info = info
            self.__cross_entropy = cross_entropy
            self.__logits = lgt
            self.__all_att_weights = att_w
            self.__caption_batch = caption_batch        
        if not is_train:
            # at the test time use beam search to generate a caption for an image:
            true_batch_size = batch_size
            decoder_initial_state = attention_cell.zero_state(
                dtype=tf.float32, batch_size=true_batch_size * beam_width)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=tiled_encoder_final_state)           
            initial_state = decoder_initial_state
            start_tokens = tf.zeros([batch_size], dtype=tf.int32)
            end_token = tf.constant(1, dtype=tf.int32) 
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                attention_cell, emb, start_tokens, end_token, initial_state, 
                beam_width, output_layer=projection_layer, length_penalty_weight=0.0, 
                coverage_penalty_weight=0.0, reorder_tensor_arrays=True)
            outputs, AttentionWrapperState, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=25)
            ids = outputs.predicted_ids
            self.__ids = ids
            self.__all_att_weights = AttentionWrapperState[0][5]
        self.__W_emb = emb

    @property
    def _train_step(self):
        return self.__train_step

    @property
    def _cross_entropy(self):
        return self.__cross_entropy
    
    @property
    def _logits(self):
        return self.__logits

    @property
    def _W_emb(self):
        return self.__W_emb

    @property
    def _grid_feat_batch(self):
        return self.__grid_feat_batch
    
    @property
    def _info(self):
        return self.__info   

    @property
    def _caption_batch(self):
        return self.__caption_batch
    
    @property
    def _ids(self):
        return self.__ids 
    
    @property
    def _scores(self):
        return self.__scores

    @property
    def _all_att_weights(self):
        return self.__all_att_weights              
###############################################################################
#######################   Grid LSTM Model   ###################################
###############################################################################                
class ICG_model_grid:
    def __init__(self, features: dict, hyperparameters: dict, is_train: bool):
        batch_size = hyperparameters['batch_size']
        grid_size = hyperparameters['grid_size']
        vocab_size = hyperparameters['vocab_size']
        word_emb_size = hyperparameters['word_emb_size']
        grid_emb_size = hyperparameters['grid_emb_size']
        grid_feat_dim = hyperparameters['grid_feat_dim'] 
        
        # Create the projection layer and the word embedding matrix:
        init =  tf.constant_initializer(0.01 * numpy.random.uniform(
            -1, 1, size=(vocab_size, word_emb_size)))
        projection_layer = Dense(vocab_size, use_bias=True, kernel_initializer=init)
        projection_layer.build((vocab_size,word_emb_size))
        emb = tf.transpose(projection_layer.trainable_weights[0]) # word_emb_size x vocab_size 
        
        # Retrieve information from hyperparameters and a batch of tf records:
        if is_train:
            dropout_keep_rate = hyperparameters['dropout_keep_rate']
            grid_feat_batch, visual_concept_batch, caption_batch, target_batch = tf.train.shuffle_batch(
                    [features['grid_feat'], features['visual_concept'], 
                     features['caption'], features['target']],
                    batch_size=batch_size, capacity=20000, min_after_dequeue=200)
            
            # The caption_batch and target_batch are sparse matrices.
            # we need to convert the to dense matrices:
            caption_batch_dense = tf.sparse_tensor_to_dense(
                sp_input=caption_batch, default_value=0)  # B x max_len
            target_batch_dense = tf.sparse_tensor_to_dense(
                sp_input=target_batch, default_value=0) # B x max_len 
            
            # embedding the ids of the captions:
            caption_batch_dense_emb = tf.nn.embedding_lookup(
                emb, caption_batch_dense) # B x max_len x word_emb_size
        else:
            # at the test time we do not have a caption, only image features
            # and visual concepts
            grid_feat_batch = features['grid_feat']
            visual_concept_batch = features['visual_concept']
            dropout_keep_rate = 1.0

        lstm_decoder_first_layer = tf.nn.rnn_cell.LSTMCell(num_units=word_emb_size)

        W_visual_concept = tf.Variable(0.01 * tf.random_normal(
            [VISUAL_CONCEPT_SIZE, word_emb_size]))
        b_visual_concept = tf.Variable(tf.zeros([word_emb_size]))
        visual_concept_proj = tf.tensordot(visual_concept_batch, W_visual_concept,
                                           [[1], [0]]) + b_visual_concept # B x word_emb_size
        visual_concept_proj = tf.reshape(visual_concept_proj,
                                         [-1, 1, word_emb_size])  # B x 1 x word_emb_size
        
        _, deocder_first_layer_init_state = tf.nn.dynamic_rnn(
            lstm_decoder_first_layer, visual_concept_proj, dtype=tf.float32)
        # init_st is B x word_emb_size
        
        self.__grid_feat_batch = grid_feat_batch # B x (grid_size * grid_size * grid_feat_dim)
        
        grid_feat_batch = tf.reshape(grid_feat_batch, [-1, grid_size * grid_size, grid_feat_dim]
                                     ) # B x (grid_size * grid_size) x grid_feat_dim
        
        # project image features into a space of dimension grid_emb_size using
        # a densely-connected layer:
        W_grid = tf.Variable(0.01 * tf.random_normal([grid_feat_dim, grid_emb_size]))
        b_grid = tf.Variable(tf.zeros([grid_emb_size]))
        feat_batch_proj = tf.tensordot(grid_feat_batch, W_grid, [[2], [0]]
                                       ) + b_grid  # B x (grid_size * grid_size) x grid_emb_size
        feat_batch_proj = tf.nn.dropout(feat_batch_proj, keep_prob=dropout_keep_rate)
        
        
        # apply a Grid LSTM to the image features
        grid_lstm_cell = grid_rnn.Grid2LSTMCell(grid_emb_size, use_peepholes=True,
                                                output_is_tuple=True, state_is_tuple=True)
        # top_left_to_bottom_right:
        grid_lstm_outputs_top_left_to_bottom_right, _ = tf.nn.dynamic_rnn(
            grid_lstm_cell, feat_batch_proj, sequence_length= grid_size * grid_size * tf.ones(
                [batch_size], dtype=tf.int32), dtype=tf.float32, scope='rnn0') 
       
        temp = tf.reshape(feat_batch_proj, [-1, grid_size, grid_size,
                              grid_emb_size])  # B x grid_size x grid_size x grid_emb_size
        # top_right_to_bottom_left:
        feat_batch_proj_rev1 = tf.reverse(temp, axis=[1])
        feat_batch_proj_rev1 = tf.reshape(
            feat_batch_proj_rev1, [-1, grid_size * grid_size, grid_emb_size])
        grid_lstm_outputs_top_right_to_bottom_left, _ = tf.nn.dynamic_rnn(
            grid_lstm_cell, feat_batch_proj_rev1, sequence_length= 
            grid_size * grid_size * tf.ones([batch_size], dtype=tf.int32),
            dtype=tf.float32, scope='rnn1')
        # bottom_left_to_top_right:
        feat_batch_proj_rev2 = tf.reverse(temp, axis=[2])
        feat_batch_proj_rev2 = tf.reshape(
            feat_batch_proj_rev2, [-1, grid_size * grid_size, grid_emb_size])      
        grid_lstm_outputs_bottom_left_to_top_right, _ = tf.nn.dynamic_rnn(
            grid_lstm_cell, feat_batch_proj_rev2, sequence_length=
            grid_size * grid_size * tf.ones([batch_size], dtype=tf.int32),
            dtype=tf.float32, scope='rnn2')
        # bottom_right_to_top_left:
        feat_batch_proj_rev3 = tf.reverse(temp, axis=[1,2])
        feat_batch_proj_rev3 = tf.reshape(
            feat_batch_proj_rev3, [-1, grid_size * grid_size, grid_emb_size])
        grid_lstm_outputs_bottom_right_to_top_left, _ = tf.nn.dynamic_rnn(
            grid_lstm_cell, feat_batch_proj_rev3, sequence_length= 
            grid_size * grid_size * tf.ones([batch_size], dtype=tf.int32),
            dtype=tf.float32, scope='rnn3')
        
        grid_lstm_outputs = grid_lstm_outputs_top_left_to_bottom_right + \
        grid_lstm_outputs_top_right_to_bottom_left + \
        grid_lstm_outputs_bottom_left_to_top_right + \
        grid_lstm_outputs_bottom_right_to_top_left
                         
        deocder_second_layer_init_state =  tf.nn.rnn_cell.LSTMStateTuple(
            c=tf.zeros([batch_size,512], dtype=tf.float32), 
            h=tf.zeros([batch_size,512], dtype=tf.float32))
        
        attention_depth = 512            
        if not is_train:
            # at the test time we need to tile the encoder_outputs:
            beam_width = 20
            encoder_outputs = grid_lstm_outputs[0]
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_width)       
            encoder_final_state = deocder_second_layer_init_state
            tiled_encoder_second_layer_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=beam_width)        
            
            sequence_length = grid_size * grid_size * tf.ones(
                [batch_size], dtype=tf.int64)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                sequence_length, multiplier=beam_width)
            encoder_final_state = deocder_first_layer_init_state               
            tiled_encoder_first_layer_final_state = tf.contrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=beam_width) 
                           
        else:
            tiled_encoder_outputs = grid_lstm_outputs[0]
            tiled_sequence_length = None 
            
        # define decoder two-layer LSTM (first layer gets the visual concepts
        # and the second layer gets the image features via an attention mechanism):
        cells = [lstm_decoder_first_layer]
        lstm_decoder_second_layer = tf.nn.rnn_cell.LSTMCell(word_emb_size)        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=attention_depth, memory=tiled_encoder_outputs,
            memory_sequence_length=tiled_sequence_length)                    
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            lstm_decoder_second_layer, attention_mechanism, alignment_history=True,
            attention_layer_size=word_emb_size)
        cells.append(attention_cell)           
        decoder_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)   
        if is_train:
            # training the decoder:
            # https://www.tensorflow.org/api_guides/python/contrib.seq2seq
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = caption_batch_dense_emb, sequence_length = 
                tf.shape(caption_batch_dense_emb)[1] * tf.ones(
                    [batch_size], dtype=tf.int32), time_major = False)
            
            decoder_initial_state2 = attention_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size * 1)
            decoder_initial_state2 = decoder_initial_state2.clone(
                cell_state= deocder_second_layer_init_state)
    
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell, helper = training_helper, initial_state =
                (deocder_first_layer_init_state, decoder_initial_state2), 
                output_layer = projection_layer)

            training_decoder_output, AttentionWrapperState, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder, impute_finished = False, maximum_iterations = 100)
            training_logits = training_decoder_output.rnn_output
            
            # the sum of the negative log likelihood of the correct word at 
            # each time step is chosen as the loss:
            lgt = training_logits            
            mask = tf.cast(target_batch_dense > 0, dtype=tf.float32)
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_batch_dense, logits=lgt)
            cost_mask = tf.multiply(mask, cost)
            cost_mask_sum = tf.reduce_sum(cost_mask, 1)
            cross_entropy = tf.reduce_mean(cost_mask_sum)
            
            # the loss is minimized:
            learning_rate = hyperparameters['learning_rate']
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  decay=0.9)           
            train_step = optimizer.minimize(cross_entropy)   
            ind_tensor = tf.range(AttentionWrapperState[1][4].size())
            att_w = AttentionWrapperState[1][4].gather(indices = ind_tensor, 
                                                       name=None)
            info = [att_w, lgt]     
            self.__train_step = train_step
            self.__info = info
            self.__cross_entropy = cross_entropy
            self.__logits = lgt
            self.__all_att_weights = att_w
            self.__caption_batch = caption_batch       
        if not is_train:
            # at the test time use beam search to generate a caption for an image:
            true_batch_size = batch_size
            decoder_initial_state = attention_cell.zero_state(
                dtype=tf.float32, batch_size=true_batch_size * beam_width)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=tiled_encoder_second_layer_final_state)                        
            initial_state = (tiled_encoder_first_layer_final_state, 
                             decoder_initial_state)
            start_tokens = tf.zeros([batch_size], dtype=tf.int32)
            end_token = tf.constant(1, dtype=tf.int32) 
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                decoder_cell, emb, start_tokens, end_token, initial_state, beam_width,
                output_layer=projection_layer, length_penalty_weight=0.0, 
                coverage_penalty_weight=0.0, reorder_tensor_arrays=True)
            outputs, AttentionWrapperState, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=25)
            ids = outputs.predicted_ids
            self.__ids = ids  
            # get the attetion weights for image regions:
            self.__all_att_weights = AttentionWrapperState[0][1][5]      
        self.__W_emb = emb
        self.__visual_concept_batch = visual_concept_batch       
    @property
    def _train_step(self):
        return self.__train_step

    @property
    def _cross_entropy(self):
        return self.__cross_entropy
    
    @property
    def _logits(self):
        return self.__logits

    @property
    def _W_emb(self):
        return self.__W_emb

    @property
    def _grid_feat_batch(self):
        return self.__grid_feat_batch
    
    @property
    def _info(self):
        return self.__info   

    @property
    def _caption_batch(self):
        return self.__caption_batch
    
    @property
    def _ids(self):
        return self.__ids 
    
    @property
    def _scores(self):
        return self.__scores

    @property
    def _visual_concept_batch(self):
        return self.__visual_concept_batch
    
    @property
    def _all_att_weights(self):
        return self.__all_att_weights    
