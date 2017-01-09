# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

from pymongo import MongoClient
from bson import ObjectId
import cPickle
import os
from collections import Counter
# This class creates a word->integer mapping or all unique words. 
# If a word is very uncommon (below threshold) it is replaced with a <UNKNOWN>-token
class WordDict:
	def __init__(self,threshold):
		client = MongoClient("mongodb://localhost:27017")
		self.document_table = client["wikipedia"]["documents"]
		self.threshold = threshold
		self.load()
	def vocab_size(self):
		return len(self.word_dict)

	def load(self):
		if os.path.isfile("word_dict_"+str(self.threshold)+".pickle"):
			self.word_dict = cPickle.load(open("word_dict_"+str(self.threshold)+".pickle","rb"))
		else:
			word_counter = Counter()
			for i,doc in enumerate(self.document_table.find({})):
				if i % 10000 == 0:
					print i
				for line in doc["text"].split("\n"):
					for word in line.split(" "):
						word_counter[word] += 1
			# Remove rare words
			words = [(word,word_counter[word]) for word in word_counter]
			words = filter(lambda x: x[1] > self.threshold,words)
			# Strip frequencies
			words = map(lambda x:x[0],words) 
			# Add special tokens
			words.append("<UNKNOWN>")
			words.append("<PAD>") 
			# Zip with indices
			self.word_dict = dict(zip(words,range(len(words))))
			cPickle.dump(self.word_dict,open("word_dict_"+str(self.threshold)+".pickle","wb"))
		self.inv_word_dict = {v: k for k, v in self.word_dict.iteritems()}
	def word2int(self,word):
		if word in self.word_dict:
			return self.word_dict[word]
		else:
			return self.word_dict["<UNKNOWN>"]
	def int2word(self,number):
		if number in self.inv_word_dict:
			return self.inv_word_dict[number]
		else:
			raise Exception("That word does not exist.")
