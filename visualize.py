# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

import tensorflow as tf 
import numpy as np 
from encoder import Encoder
from train_data_batcher import TrainDataBatcher
from word_dict import WordDict
from time import time
import cPickle
from random import randint
import random


class Visualize:
	def __init__(self,max_summary_words):
		self.max_summary_words = max_summary_words
		word_dict = WordDict(threshold=5)
		self.data_batcher = TrainDataBatcher(word_dict)
		vocab_size = word_dict.vocab_size()
		# Placeholders:
		self.summary_index_placeholder = tf.placeholder(tf.int32,shape=[None,self.max_summary_words])
		self.dropout_placeholder = tf.placeholder(tf.float32)
		# Models
		self.sess = tf.Session()
		self.encoder = Encoder(vocab_size,embedding_size=128,dropout_placeholder=self.dropout_placeholder)
		self.vector = tf.clip_by_norm(self.encoder.encode(self.summary_index_placeholder,code_size=300),3.0,axes=1)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess,"pretrained/encoder.ckpt")

	def save_vector_cache(self):
		title_list = []
		matricies = []
		saved_files = 0 
		for i,(title_batch,summary_batch) in enumerate(self.data_batcher.iter_summaries(1000,self.max_summary_words)):
			print i,")"
			vectors = self.sess.run(self.vector,feed_dict={self.summary_index_placeholder:summary_batch,self.dropout_placeholder:1.0})
			title_list += title_batch
			matricies.append(vectors)
			if len(matricies) == 50:
				with open("vector_cache/"+str(saved_files)+"_matrix.pickle","wb") as f:
					cPickle.dump(np.vstack(matricies),f)
				with open("vector_cache/"+str(saved_files)+"_titles.pickle","wb") as f:
					cPickle.dump(title_list,f)
				matricies = []
				title_list = []
				saved_files += 1
		with open("vector_cache/"+str(i)+"_matrix.pickle","wb") as f:
			cPickle.dump(np.vstack(matricies),f)
		with open("vector_cache/"+str(i)+"_titles.pickle","wb") as f:
			cPickle.dump(title_list,f)	

	def search(self):
		while True:
			i = randint(0,60-1)
			with open("vector_cache/"+str(i)+"_matrix.pickle","rb") as f:
				matrix = cPickle.load(f)
			with open("vector_cache/"+str(i)+"_titles.pickle","rb") as f:
				titles = cPickle.load(f)
			idx = randint(0,len(titles)-1)
			vector = matrix[idx]
			title = titles[idx]
			
			neighbors = []


			for i in range(60):
				with open("vector_cache/"+str(i)+"_matrix.pickle","rb") as f:
					matrix = cPickle.load(f)
				with open("vector_cache/"+str(i)+"_titles.pickle","rb") as f:
					titles = cPickle.load(f)
				best =  np.argpartition(-np.dot(matrix,np.transpose(vector)),5)[:5]
				
				
				for b in best:
					if titles[b] != title:
						score = np.dot(vector,matrix[b])
						neighbors.append((score,titles[b]))
				
			print "RESULT==="
			print title
			print "***"
			s = sorted(neighbors,key=lambda x:-x[0])[:5]
			for score,name in s:
				print score,name
			print "\n====\n"
vis = Visualize(max_summary_words=100)
vis.search()
