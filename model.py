# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as tfslim
from discriminator import Discriminator
from encoder import Encoder
from document_batcher import DocumentBatcher
from data_set import DataSet
from time import time
import sys
import requests

from boto.s3.connection import S3Connection
from boto.s3.key import Key

class Model:
	def __init__(self,max_query_words,max_summary_words):
		self.data_set = DataSet()
		self.data_set.load("normal_ds")
		self.pad_symbol = self.data_set.pad_symbol()
		self.data_batcher = DocumentBatcher()
		self.batch_size = 32
		self.documents_per_query = 2
		self.embedding_size = 128
		self.vocab_size = self.data_set.vocab_size()
		self.max_query_words = max_query_words
		self.max_summary_words = max_summary_words
		self.embedding_learning_scaler = 5.0
		# Placeholders:
		self.query_index_placeholder = tf.placeholder(tf.int32,shape=[None,None])
		self.summary_index_placeholder = tf.placeholder(tf.int32,shape=[None,None])
		self.output_placeholder = tf.placeholder(tf.float32,shape=[None])
		self.dropout_placeholder = tf.placeholder(tf.float32)
		# Models
		self.encoder = Encoder(self.vocab_size,self.embedding_size,dropout_placeholder=self.dropout_placeholder)
		self.discriminator = Discriminator()

		# Forward pass 
		self.query_vector = tf.clip_by_norm(self.encoder.encode_cnn(self.query_index_placeholder,max_text_length=max_query_words,code_size=300),3.0,axes=1)
		self.summary_vector = tf.clip_by_norm(self.encoder.encode_cnn(self.summary_index_placeholder,max_text_length=max_summary_words,code_size=300),3.0,axes=1)
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

		self.gradients = gradients
		self.train_step = optimizer.apply_gradients(gradients)
		self.sess = tf.Session()
		tf.initialize_all_variables().run(session=self.sess)
		self.saver = tf.train.Saver()
		# Stuff needed to save the model to AWS S3:
		conn = None#S3Connection()
		self.bucket = None#conn.get_bucket("burenoreus-machinelearning")
	# Save model to disk
	def save(self):
		t1 = time()
		self.saver.save(self.sess,"models/encoder.ckpt")
		print "TIME TO SAVE MODEL",time()-t1

	def load(self):
		t1 = time()
		print "load:"
		self.saver.restore(self.sess,"models/encoder.ckpt")
		print "TIME TO LOAD MODEL ",time()-t1
	def test(self):
		test_loss = 0.0
		test_count = 0.0
		t1 = time()
		for i,(summary_batch,query_batch,response_batch) in enumerate(self.data_batcher.batch(self.data_set.iter_test,self.documents_per_query,1000)):
			summary_batch = self.data_batcher.pad_batch(summary_batch,self.pad_symbol,100)
			query_batch = self.data_batcher.pad_batch(query_batch,self.pad_symbol,100)
			feed_dict = {self.output_placeholder:response_batch,self.summary_index_placeholder:summary_batch,self.query_index_placeholder:query_batch,self.dropout_placeholder:1.0}
			test_loss += len(summary_batch)*self.sess.run(self.logloss,feed_dict)
			test_count += len(summary_batch)
		print "=========================================="
		print "Validation result: ",test_loss/test_count , " Time elapsed:",time()-t1
		print "=========================================="
		return test_loss/test_count
	def train(self):
		#self.load()
		#sys.exit()
		validation_summary = []
		validation_query = []
		validation_response = []
		train_loss = 0.0
		train_count = 0.0
		t1 = time()
		best_test_loss = 99999.0
		validation_log = open("validation_log.txt","w")
		validation_log.write("Validation errors:")
		for epoch in range(100):
			for i,(summary_batch,query_batch,response_batch) in enumerate(self.data_batcher.batch(self.data_set.iter_train,self.documents_per_query,self.batch_size)):
				summary_batch = self.data_batcher.pad_batch(summary_batch,self.pad_symbol,100)
				query_batch = self.data_batcher.pad_batch(query_batch,self.pad_symbol,100)
				feed_dict = {self.output_placeholder:response_batch,self.summary_index_placeholder:summary_batch,self.query_index_placeholder:query_batch,self.dropout_placeholder:0.8}
				_,loss = self.sess.run([self.train_step,self.logloss],feed_dict)
	
				train_count += 1.0
				train_loss += loss

				if i % 150 == 0:
					pass#self.check_spot_termination()

				# Calculate a training error as we go along
				if i % 1000 == 0:
					print "Epoch ",epoch, " Train step ",i, " training loss=",train_loss/train_count, " Time elapsed: ",time()-t1
					#print "Output "," ".join(map(str,list(self.sess.run(self.discriminator.o,feed_dict))))
					train_loss = 0.0
					train_count = 0.0
					t1 = time()
				# Sometimes save the model and validate progress
				if i % 30000 == 0:
					
					test_loss = self.test()
					if test_loss < best_test_loss:
						self.save()
					else:
						print "WE DIDN'T IMPROVE VALIDATION LOSS. DO NOT SAVE THAT MODEL"
					validation_log.write(str(test_loss)+"\n")
		validation_log.close()

	def check_spot_termination(self):
		try:
			r = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time")
			if r.status_code != 404:
				self.save()
				self.upload_to_s3()
				print "Exiting..."
				sys.exit()
		except:
			pass
	
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

model = Model(max_query_words=100,max_summary_words=100)
model.train()