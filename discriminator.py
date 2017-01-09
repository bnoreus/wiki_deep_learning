# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.slim as tfslim
from encoder import Encoder

class Discriminator:
	def __init__(self):
		pass
	# Takes the word indices for a batch of queries and a batch of summaries and converts to output probability
	def predict(self,query_vector,summary_vector):
		# Architecture config:
		# Forward pass:
		self.b = tf.get_variable("B",shape=[1])
		#concat = tf.concat(1,[query_vector,summary_vector])
		#print concat.get_shape()
		#hidden_1 = tfslim.fully_connected(concat,hidden_1_size,activation_fn=tf.nn.relu,weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
		#output = tfslim.fully_connected(hidden_1,1,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.001))
		output = tf.reduce_sum(tf.mul(query_vector,summary_vector),axis=1)
		#output = tf.clip_by_value(output,-5.0,5.0)
		self.o = output
		prob = tf.sigmoid(output)
		return prob

