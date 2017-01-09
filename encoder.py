# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

import tensorflow as tf 
import numpy as np 
from boto.s3.connection import S3Connection
from boto.s3.key import Key

from os import listdir
from os.path import isfile, join
from time import time

class Encoder:
	def __init__(self,vocabulary_size,embedding_size,max_query_words,max_summary_words,dropout_placeholder):
		self.word_embedding = tf.get_variable("word_embedding",shape=[vocabulary_size,embedding_size])
		self.max_summary_words = max_summary_words
		self.max_query_words = max_query_words
		self.embedding_size = embedding_size
		self.dropout_placeholder = dropout_placeholder
		self.reuse=None

		# Stuff needed to save the model to S3:
		conn = S3Connection()
		self.bucket = conn.get_bucket("burenoreus-machinelearning")
	def encode_query(self,index_placeholder):
		# Architecture config:
		hidden_1_size = 4
		hidden_2_size = 4
		# Forward pass:
		emb = tf.nn.embedding_lookup(self.word_embedding,index_placeholder)
		volume = tf.expand_dims(emb,3) #Turn into 3D conv volume
		hidden_1 = tfslim.conv2d(volume,hidden_1_size,[2,self.embedding_size],stride=2,padding="VALID",activation_fn=tf.nn.relu)
		hidden_2 = tfslim.conv2d(hidden_1,hidden_2_size,[2,1],stride=2,padding="VALID",activation_fn=None)

		hidden_2_flat = tf.reshape(hidden_2,[-1,self.max_summary_words*hidden_1_size/2/2])
		return hidden_2_flat

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

					print "****** FILTER SIZE ",filter_size," *******"
					print "Embedding_layer_shape",embedding_layer.get_shape()

					conv = tf.nn.conv2d(embedding_layer,conv_w,strides=[1,1,1,1],padding="VALID",name="conv")
					print "Conv layer shape",conv.get_shape()
					hidden_1 = tf.nn.relu(tf.nn.bias_add(conv,conv_b))

					pooled = tf.nn.max_pool(hidden_1,ksize=[1,max_text_length-(filter_size-1),1,1],strides=[1,1,1,1],padding="VALID",name="pool")
					print "Pooled layer shape",pooled.get_shape()
					pooled_outputs.append(pooled)
			concat_pooled = tf.concat(3,pooled_outputs)
			print "Concat pool shape=",concat_pooled.get_shape()
			concat_pooled = tf.reshape(concat_pooled,[-1,tot_num_filters])
			print "Concat FLAT pool shape=",concat_pooled.get_shape()
			concat_pooled = tf.nn.dropout(concat_pooled,self.dropout_placeholder)

			w_out = tf.get_variable("w_out",shape=[tot_num_filters,code_size],initializer=tf.contrib.layers.xavier_initializer())
			b_out = tf.get_variable("b_out",shape=[code_size],initializer=tf.contrib.layers.xavier_initializer())
			output_layer = tf.matmul(concat_pooled,w_out)+b_out
			self.reuse = True
			return output_layer
	# Reshapes input of size [batch_size,number_of_words,embedding_size] to [batch_size,number_of_words*embedding_size]
	def flatten_embedding(self,emb,max_number_words):
		batch_size = tf.reduce_sum(tf.slice(tf.shape(emb),[0],[1])) 
		shape = [batch_size,max_number_words*self.embedding_size]
		return tf.reshape(emb,shape)

	# Save model to disk
	def save(self,session):
		t1 = time()
		saver = tf.train.Saver()
		with tf.variable_scope("encoder",reuse=True):
			saver.save(session,"models/encoder.ckpt")
		print "TIME TO SAVE MODEL",time()-t1

	# Save model to AWS ec2	
	# You need to set the following environment variables for this to work:
	# AWS_ACCESS_KEY_ID - Your AWS Access Key ID
	# AWS_SECRET_ACCESS_KEY - Your AWS Secret Access Key
	# Use this function after using save() described above.
	def upload_to_s3(self):
		files = [f for f in listdir("models") if isfile(join("models", f))]
		for file in files:
			k = Key(self.bucket)
			k.key = file
			k.set_contents_from_filename("models/"+file)

