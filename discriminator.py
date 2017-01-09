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
		output = tf.reduce_sum(tf.mul(query_vector,summary_vector),axis=1)
		self.o = output
		prob = tf.sigmoid(output)
		return prob

