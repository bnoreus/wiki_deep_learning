# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as tfslim
from discriminator import Discriminator
from encoder import Encoder
from train_data_batcher import TrainDataBatcher
from word_dict import WordDict
from time import time
import sys
class Model:
	def __init__(self,max_query_words,max_summary_words):
		word_dict = WordDict(threshold=5)
		self.data_batcher = TrainDataBatcher(word_dict)
		self.batch_size = 32
		self.documents_per_query = 2
		self.embedding_size = 128
		self.vocab_size = word_dict.vocab_size()
		self.max_query_words = max_query_words
		self.max_summary_words = max_summary_words
		self.embedding_learning_scaler = 5.0
		# Placeholders:
		self.query_index_placeholder = tf.placeholder(tf.int32,shape=[None,max_query_words])
		self.summary_index_placeholder = tf.placeholder(tf.int32,shape=[None,max_summary_words])
		self.output_placeholder = tf.placeholder(tf.float32,shape=[None])
		self.dropout_placeholder = tf.placeholder(tf.float32)
		# Models
		self.encoder = Encoder(self.vocab_size,self.embedding_size,max_query_words=max_query_words,max_summary_words=max_summary_words,dropout_placeholder=self.dropout_placeholder)
		self.discriminator = Discriminator()

		# Forward pass 
		self.query_vector = tf.clip_by_norm(self.encoder.encode(self.query_index_placeholder,code_size=300),3.0,axes=1)
		self.summary_vector = tf.clip_by_norm(self.encoder.encode(self.summary_index_placeholder,code_size=300),3.0,axes=1)
		self.prediction = self.discriminator.predict(self.query_vector,self.summary_vector)

		# Optimization
		optimizer = tf.train.AdagradOptimizer(0.1)
		self.logloss = -tf.reduce_mean(self.output_placeholder*tf.log(self.prediction)+(1.0-self.output_placeholder)*tf.log(1.0-self.prediction))
		gradients = optimizer.compute_gradients(self.logloss)
		for i,(grad,var) in enumerate(gradients):
			if grad is not None:
				if "word_embedding" in var.name:
					s = tf.IndexedSlices(indices=grad.indices,values=grad.values*self.embedding_learning_scaler,dense_shape=grad.dense_shape)
					gradients[i] = (s,var)
			print gradients[i]

		self.gradients = gradients
		self.train_step = optimizer.apply_gradients(gradients)
		self.sess = tf.Session()
		tf.initialize_all_variables().run(session=self.sess)

	def test(self):
		print "Validating..."
		test_loss = 0.0
		test_count = 0.0
		t1 = time()
		for i,(summary_batch,query_batch,response_batch) in enumerate(self.data_batcher.mini_batch_from_cache(1000,"../wikitest.csv")):
			if i % 10 == 0:
				print "Validated ",1000*i, " rows"
			feed_dict = {self.output_placeholder:response_batch,self.summary_index_placeholder:summary_batch,self.query_index_placeholder:query_batch,self.dropout_placeholder:1.0}
			test_loss += len(summary_batch)*self.sess.run(self.logloss,feed_dict)
			test_count += len(summary_batch)
		print "Validation result: ",test_loss/test_count , " Time elapsed:",time()-t1

	def train(self):
		validation_summary = []
		validation_query = []
		validation_response = []
		rows_trained = 0
		train_loss = 0.0
		train_count = 0.0
		t1 = time()
		for i,(summary_batch,query_batch,response_batch) in enumerate(self.data_batcher.mini_batch_from_cache(self.batch_size,"../wikitrain.csv")):
			feed_dict = {self.output_placeholder:response_batch,self.summary_index_placeholder:summary_batch,self.query_index_placeholder:query_batch,self.dropout_placeholder:1.0}
			_,loss = self.sess.run([self.train_step,self.logloss],feed_dict)
			train_count += 1.0
			train_loss += loss
	
			if i % 100 == 0 and i > 0:
				print "Train step ",i, " training loss=",train_loss/train_count, " Time elapsed: ",time()-t1
				#print "Output "," ".join(map(str,list(self.sess.run(self.discriminator.o,feed_dict))))
				train_loss = 0.0
				train_count = 0.0
				t1 = time()
			if i % 10000 == 0 and i > 0:
				self.encoder.save(self.sess)
			if i % 100000 == 0:
				self.test()
model = Model(max_query_words=100,max_summary_words=100)
model.train()