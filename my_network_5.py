import tensorflow as tf
import numpy as np

'''
Multi-task with type prediction. square of PR curve is about 0.381.
Drop word level attention, select entity word for relation classification.
'''


class Settings(object):

	def __init__(self):

		self.vocab_size = 114042
		self.num_steps = 70
		self.num_epochs = 3
		self.num_classes = 53
		self.gru_size = 230
		self.keep_prob = 0.5
		self.num_layers = 1
		self.pos_size = 5
		self.pos_num = 123
		# the number of entity pairs of each batch during training or testing
		self.big_num = 50
		
		self.embedding_dim = 50
		self.type_head = np.load('./data/type_head.npy')
		self.type_tail = np.load('./data/type_tail.npy')
		

class GRU:
	def __init__(self,is_training,word_embeddings,settings):
		
		self.num_steps = num_steps = settings.num_steps	
		self.vocab_size = vocab_size = settings.vocab_size
		self.num_classes = num_classes = settings.num_classes
		self.gru_size = gru_size = settings.gru_size
		self.big_num = big_num = settings.big_num
		self.type_head = settings.type_head
		self.type_tail = settings.type_tail
		self.embedding_dim = settings.embedding_dim

		self.input_word = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='input_word')
		self.input_pos1 = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='input_pos1')
		self.input_pos2 = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='input_pos2')
		self.absolute_pos1 = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='absolute_pos1')
		self.absolute_pos2 = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='absolute_pos2')
		self.input_y = tf.placeholder(dtype=tf.float32,shape=[None,num_classes],name='input_y')
		self.total_shape = tf.placeholder(dtype=tf.int32,shape=[big_num+1],name='total_shape')
		total_num = self.total_shape[-1]

		word_embedding = tf.get_variable(initializer=word_embeddings,name = 'word_embedding')
		pos1_embedding = tf.get_variable('pos1_embedding',[settings.pos_num,settings.pos_size])
		pos2_embedding = tf.get_variable('pos2_embedding',[settings.pos_num,settings.pos_size])

		attention_w = tf.get_variable('attention_omega',[gru_size,1])
		sen_a = tf.get_variable('attention_A',[gru_size*2])
		sen_r = tf.get_variable('query_r',[gru_size*2,1])		
		#relation_embedding = tf.get_variable('relation_embedding',[self.num_classes,gru_size*2])
		relation_embedding = tf.get_variable('relation_embedding',[self.embedding_dim*4,gru_size*2])
		sen_d = tf.get_variable('bias_d',[self.num_classes])

                sen_a_r = tf.get_variable('attention',[gru_size*2])
                sen_r_r = tf.get_variable('query',[gru_size*2,1])
                #relation_embedding = tf.get_variable('relation_embedding',[self.num_classes,gru_size*2])
                relation_embedding_r = tf.get_variable('relation_embedding_r',[self.num_classes, gru_size*2])
                sen_d_r = tf.get_variable('bias_d_r',[self.num_classes])

		gru_cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
		gru_cell_backward = tf.nn.rnn_cell.GRUCell(gru_size)


		if is_training and settings.keep_prob < 1:
			gru_cell_forward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_forward,output_keep_prob=settings.keep_prob)
			gru_cell_backward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_backward,output_keep_prob=settings.keep_prob)
		
		cell_forward = tf.nn.rnn_cell.MultiRNNCell([gru_cell_forward]*settings.num_layers)
		cell_backward = tf.nn.rnn_cell.MultiRNNCell([gru_cell_backward]*settings.num_layers)		

		sen_repre_type = []
		sen_alpha_type = []
		sen_s_type = []
		sen_out_type = []
                sen_repre_r = []
                sen_alpha_r = []
                sen_s_r = []
                sen_out_r = []
		self.prob = []
		self.predictions = []
		self.loss = []
		self.accuracy = []
		self.total_loss = 0.0
		
		self._initial_state_forward = cell_forward.zero_state(total_num,tf.float32)
		self._initial_state_backward = cell_backward.zero_state(total_num,tf.float32)

		# embedding layer
		inputs_forward = tf.concat(2,[tf.nn.embedding_lookup(word_embedding,self.input_word),tf.nn.embedding_lookup(pos1_embedding,self.input_pos1),tf.nn.embedding_lookup(pos2_embedding,self.input_pos2)])
		inputs_backward = tf.concat(2,[tf.nn.embedding_lookup(word_embedding,tf.reverse(self.input_word,[False,True])),tf.nn.embedding_lookup(pos1_embedding,tf.reverse(self.input_pos1,[False,True])),tf.nn.embedding_lookup(pos1_embedding,tf.reverse(self.input_pos2,[False,True]))])
		
		relation_type = []
		relation_NA = tf.concat(1,[np.random.normal(size=[1,self.embedding_dim*4],loc=0,scale=0.05).astype('float32')])
		relation_type.append(relation_NA)
		for i in range(len(self.type_head)):
			head = self.type_head[i].strip().split()
			head_t1 = head[0].split('_')
			head_t2 = head[1].split('_')
			temp_embedding_h1 = np.zeros(self.embedding_dim, dtype=np.float32)
			temp_embedding_h2 = np.zeros(self.embedding_dim, dtype=np.float32)
			for j in range(len(head_t1)):
				temp_embedding_h1 += tf.nn.embedding_lookup(word_embedding, np.array([int(head_t1[j])]))
			temp_embedding_h1 /= len(head_t1)
			for j in range(len(head_t2)):
                                temp_embedding_h2 += tf.nn.embedding_lookup(word_embedding, np.array([int(head_t2[j])]))
                        temp_embedding_h2 /= len(head_t1)
			temp_embedding_head = tf.concat(1, [temp_embedding_h1, temp_embedding_h2])

                        tail = self.type_head[i].strip().split()
                        tail_t1 = tail[0].split('_')
                        tail_t2 = tail[1].split('_')
			temp_embedding_t1 = np.zeros(self.embedding_dim, dtype=np.float32)
			temp_embedding_t2 = np.zeros(self.embedding_dim, dtype=np.float32)

                        for j in range(len(tail_t1)):
                                temp_embedding_t1 += tf.nn.embedding_lookup(word_embedding, np.array([int(tail_t1[j])]))
                        temp_embedding_h1 /= len(tail_t1)
                        for j in range(len(tail_t2)):
                                temp_embedding_t2 += tf.nn.embedding_lookup(word_embedding, np.array([int(tail_t2[j])]))
                        temp_embedding_t2 /= len(tail_t2)
                        temp_embedding_tail = tf.concat(1, [temp_embedding_t1, temp_embedding_t2])

			relation_type.append(tf.concat(1, [temp_embedding_head, temp_embedding_tail]))
 			
		relation_type = np.array(relation_type)
		rels = relation_type[0]
		for i in range(len(relation_type)-1):
			rels = tf.concat(0,[rels,relation_type[i+1]])			
		#print 'relation types embeddings: %s' % rels

		outputs_forward = []

		state_forward = self._initial_state_forward

		# Bi-GRU layer
		with tf.variable_scope('GRU_FORWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_forward,state_forward) = cell_forward(inputs_forward[:,step,:],state_forward)
				outputs_forward.append(cell_output_forward)				
		
		outputs_backward = []

		state_backward = self._initial_state_backward
		with tf.variable_scope('GRU_BACKWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_backward,state_backward) = cell_backward(inputs_backward[:,step,:],state_backward)
				outputs_backward.append(cell_output_backward)
		
		output_forward = tf.reshape(tf.concat(1,  outputs_forward), [total_num, num_steps, gru_size])
		output_backward  = tf.reverse(tf.reshape(tf.concat(1,  outputs_backward), [total_num, num_steps, gru_size]), [False, True, False])
		
		# word-level attention layer
		output_h = tf.add(output_forward,output_backward)
		attention_r = tf.reshape(tf.batch_matmul(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h),[total_num*num_steps,gru_size]),attention_w),[total_num,num_steps])),[total_num,1,num_steps]),output_h),[total_num,gru_size])
		
		attention_h = tf.reshape(tf.batch_matmul(tf.reshape(tf.cast(self.absolute_pos1, tf.float32),[total_num,1,num_steps]),output_h),[total_num, gru_size])
		attention_t = tf.reshape(tf.batch_matmul(tf.reshape(tf.cast(self.absolute_pos2, tf.float32),[total_num,1,num_steps]),output_h),[total_num, gru_size])
		
		attention_type = tf.concat(1 , [attention_h, attention_t])
		# sentence-level attention layer
		for i in range(big_num):

			sen_repre_type.append(tf.tanh(attention_type[self.total_shape[i]:self.total_shape[i+1]]))
			sen_repre_r.append(tf.tanh(attention_type[self.total_shape[i]:self.total_shape[i+1]]))

			batch_size = self.total_shape[i+1]-self.total_shape[i]
		
			sen_alpha_type.append(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.mul(sen_repre_type[i],sen_a),sen_r),[batch_size])),[1,batch_size]))
			sen_alpha_r.append(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.mul(sen_repre_r[i],sen_a_r),sen_r_r),[batch_size])),[1,batch_size]))
		
			sen_s_type.append(tf.reshape(tf.matmul(sen_alpha_type[i],sen_repre_type[i]),[gru_size*2,1]))
			sen_s_r.append(tf.reshape(tf.matmul(sen_alpha_r[i],sen_repre_r[i]),[gru_size*2,1]))
			sen_out_type.append(tf.add(tf.reshape(tf.matmul(tf.reshape(tf.matmul(rels,relation_embedding),[self.num_classes, gru_size*2]),sen_s_type[i]),[self.num_classes]),sen_d))
			sen_out_r.append(tf.add(tf.reshape(tf.matmul(relation_embedding_r,sen_s_r[i]),[self.num_classes]),sen_d_r))
			
			self.prob.append(tf.nn.softmax(sen_out_r[i]))

			with tf.name_scope("output"):
				self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

			with tf.name_scope("loss"):
				self.loss.append(0.5*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(sen_out_r[i], self.input_y[i]))+0.5*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(sen_out_type[i], self.input_y[i])))
				if i == 0:
					self.total_loss = self.loss[i]
				else:
					self.total_loss += self.loss[i]

			#tf.summary.scalar('loss',self.total_loss)
			#tf.scalar_summary(['loss'],[self.total_loss])
			with tf.name_scope("accuracy"):
				self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"), name="accuracy"))

		#tf.summary.scalar('loss',self.total_loss)
		tf.scalar_summary('loss',self.total_loss)
		#regularization
		self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())
		self.final_loss = self.total_loss+self.l2_loss
		tf.scalar_summary('l2_loss',self.l2_loss)
		tf.scalar_summary('final_loss',self.final_loss)

