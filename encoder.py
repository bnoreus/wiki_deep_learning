# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

import tensorflow as tf 
import numpy as np 


from os import listdir
from os.path import isfile, join
from time import time

class Encoder:
	def __init__(self,vocabulary_size,embedding_size,dropout_placeholder):
		self.word_embedding = tf.get_variable("word_embedding",shape=[vocabulary_size,embedding_size])
		self.embedding_size = embedding_size
		self.dropout_placeholder = dropout_placeholder
		self.reuse=None



	def encode(self,index_placeholder,code_size=100):
		filter_sizes = [2,3,4,5]
		num_filters = 100

		max_text_length = index_placeholder.get_shape()[1]
		tot_num_filters = num_filters*len(filter_sizes)
		#Embedding layer:
		embedding_layer = tf.nn.embedding_lookup(self.word_embedding,index_placeholder)
		embedding_layer = tf.expand_dims(embedding_layer,3)
		
		#Forward pass:
		with tf.variable_scope("encoder", reuse=self.reuse): #reuse the second time
			pooled_outputs = []
			for i, filter_size in enumerate(filter_sizes):
				with tf.variable_scope("conv-maxpool-%s" % filter_size):
					# Variables in this filter size:
					conv_w = tf.get_variable("conv_w",[filter_size,self.embedding_size,1,num_filters])
					conv_b = tf.get_variable("conv_b",[num_filters])
					conv = tf.nn.conv2d(embedding_layer,conv_w,strides=[1,1,1,1],padding="VALID",name="conv")
					hidden_1 = tf.nn.relu(tf.nn.bias_add(conv,conv_b))

					pooled = tf.nn.max_pool(hidden_1,ksize=[1,max_text_length-(filter_size-1),1,1],strides=[1,1,1,1],padding="VALID",name="pool")
					pooled_outputs.append(pooled)
			concat_pooled = tf.concat(3,pooled_outputs)
			concat_pooled = tf.reshape(concat_pooled,[-1,tot_num_filters])
			concat_pooled = tf.nn.dropout(concat_pooled,self.dropout_placeholder)
			w_out = tf.get_variable("w_out",shape=[tot_num_filters,code_size],initializer=tf.contrib.layers.xavier_initializer())
			b_out = tf.get_variable("b_out",shape=[code_size],initializer=tf.contrib.layers.xavier_initializer())
			output_layer = tf.matmul(concat_pooled,w_out)+b_out
			self.reuse = True
			return output_layer
			
	
