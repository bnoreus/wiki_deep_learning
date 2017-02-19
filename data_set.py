# This Python file uses the following encoding: utf-8
# Author: Bure Noreus
import time
import os
import mmap
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
import cPickle
from collections import Counter

class DataSet:
	def __init__(self):
		self.loaded = False
		self.word_dict = None
	def create(self,config):
		if self.loaded:
			raise Exception("This dataset has already been loaded!")
		if os.path.isdir("../"+config["name"]):
			raise Exception("This dataset is already created!")
		os.mkdir("../"+config["name"])
		self.name = config["name"]
		self.split_ratio = config["split_ratio"]
		self.threshold = config["word_dict_threshold"]
		self.word_dict = self.build_word_dict()
		self.build_word_csv()
		with open("../"+self.name+"/config.pickle","wb") as f:
			cPickle.dump(config,f)

	def pad_symbol(self):
		return self.word_dict["<PAD>"]

	def load(self,name):
		if self.loaded:
			raise Exception("This dataset has already been loaded!")
		self.name = name
		with open("../"+str(self.name)+"/word_dict.pickle","rb") as f:
			self.word_dict = cPickle.load(f)
		self.loaded = True
	def vocab_size(self):
		if not self.loaded:
			raise Exception("This dataset is not loaded")
		return len(self.word_dict)
	def iter_csv(self,path):
		if not self.loaded:
			raise Exception("This data set is not loaded")
		for line in open(path):
			paragraphs = line.strip().split("_")
			paragraphs = [map(int,p.split(" ")) for p in paragraphs]
			yield paragraphs

	def iter_test(self):
		for x in self.iter_csv("../"+self.name+"/test_file.csv"):
			yield x
	def iter_train(self):
		for x in self.iter_csv("../"+self.name+"/train_file.csv"):
			yield x

	def tot_number_documents(self):
		client = MongoClient("mongodb://localhost:27017")
		document_table = client["wikipedia"]["documents"]
		return document_table.count()
	def iter_mongo_docs(self):
		client = MongoClient("mongodb://localhost:27017")
		document_table = client["wikipedia"]["documents"]
		for doc in document_table.find({}):
			yield doc["title"],doc["text"]

	def build_word_dict(self):
		word_counter = Counter()
		print "Building word dict.."
		for t,(title,text) in enumerate(self.iter_mongo_docs()):
			if t % 1000 == 0:
				print t," documents scanned."
			for line in text.split("\n"):
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
		word_dict = dict(zip(words,range(0,len(words))))
		with open("../"+str(self.name)+"/word_dict.pickle","wb") as f:
			cPickle.dump(word_dict,f)
		return word_dict
	def word2int(self,word):
		if word in self.word_dict:
			return self.word_dict[word]
		else:
			return self.word_dict["<UNKNOWN>"]
	def build_word_csv(self):
		data_size = self.tot_number_documents()
		train_file = open("../"+self.name+"/train_file.csv","w")
		test_file = open("../"+self.name+"/test_file.csv","w")
		for t,(title,text) in enumerate(self.iter_mongo_docs()):
			paragraphs = []
			if t % 1000 == 0:
				print t," documents scanned"
			for line in text.split("\n"):
				word_indices = map(self.word2int,line.split(" "))
				word_indices_str = " ".join(map(str,word_indices))
				paragraphs.append(word_indices_str)
			paragraphs = "_".join(paragraphs)
			if t < data_size*self.split_ratio:
				train_file.write(paragraphs+"\n")
			else:
				test_file.write(paragraphs+"\n")
		train_file.close()
		test_file.close()
