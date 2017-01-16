# This Python file uses the following encoding: utf-8
# Author: Bure Noreus
from document_batcher import DocumentBatcher
from word_dict import WordDict
import time
import os
import mmap
import numpy as np

class TrainDataBatcher:
	def __init__(self,word_dict):
		self.batcher = DocumentBatcher()
		self.word_dict = word_dict
		self.offsets = None
		self.cache_path = "../wikitrain.csv"
		self.scan_offsets()
	# Creates a cache of the original MongoDB. The cache is very memory efficient because it stores word indices, not words. 
	# In order to recover the words, the original WordDict object is needed.
	def create_cache(self):
		with open("../wiki_cache.csv","w") as file:
			for batch_idx,(summary_batch,query_batch,response_batch) in enumerate(self.mini_batch(500,2,100,100)):
				if batch_idx % 1000 == 0:
					print "Saved ",batch_idx*500, " rows of data"
				for i in range(500):
					summary = summary_batch[i]
					query = query_batch[i]
					response = response_batch[i]
					file.write(" ".join(map(str,summary))+","+" ".join(map(str,query))+","+str(response)+"\n")


	def scan_offsets(self):
		tic = time.time()
		tmp_offsets = []  # python auto-extends this
		self.filesize = int(os.stat(self.cache_path).st_size)
		print("Scanning file '%s' to find byte offsets for each line..." % self.cache_path)
		with open(self.cache_path) as f:
			i = 0  # technically, this can grow unbounded...practically, not an issue
			mm = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)  # lazy eval-on-demand via POSIX filesystem
			for line in iter(mm.readline, ''):
				pos = mm.tell()
				tmp_offsets.append(pos)
				i += 1
				if i % int(1e6) == 0:
					print("%dM examples scanned" % (i / 1e6))
		toc = time.time()
		offsets = np.asarray(tmp_offsets, dtype = 'uint64')  # convert to numpy array for compactness; can use uint32 for small and medium corpora (i.e., less than 100M lines)
		del tmp_offsets  # don't need this any longer, save memory
		print("...file has %d bytes and %d lines" % (self.filesize, i))
		print("%.2f seconds elapsed scanning file for offsets" % (toc - tic))
		self.offsets = offsets

	def iter_train(self):
		with open(self.cache_path, 'r+b') as f:
			mm = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)
			filesize = os.stat(self.cache_path).st_size  # get filesize
			len_offsets = len(self.offsets)  # compute once
			for line_number, offset in enumerate(self.offsets):  # traverse random path
				if int(line_number) >= len_offsets:
					print("Error at line number: %d" % line_number)
					continue
				offset_begin = self.offsets[line_number]
				try:
					mm.seek(offset_begin)
					line = mm.readline()
				except:
					print("Error at location: %d" % offset)
					continue
				if len(line) == 0:
					continue	# no point to returning an empty list (i.e., whitespace)
				yield line # chain parsing logic/functions here

	def iter_test(self):
		for line in open("../wikitest.csv"):
			yield line
	# Retreive a data batch from a data cache
	# This assumes that you have created a cache CSV file with the function create_cache() described above.
	# I also recommend create a cache file and splitting that file randomly into a test a train set.
	def mini_batch_from_cache(self,batch_size,iter_func):
		summary_batch = []
		query_batch = []
		response_batch = []
		for line in iter_func():
			fields = line.strip().split(",")
			summary = fields[0].split(" ")
			query = fields[1].split(" ")
			response = float(fields[2])
			summary_batch.append(summary)
			query_batch.append(query)
			response_batch.append(response)
			if len(summary_batch)==batch_size:
				yield summary_batch,query_batch,response_batch
				summary_batch = []
				query_batch = []
				response_batch = []
		yield summary_batch,query_batch,response_batch
				
	# Retrieve a mini batch generator of [batch_size] of document pairs of a certain dilution. Dilution is the number of documents per query.
	def mini_batch(self,batch_size,documents_per_query,max_summary_words,max_query_words):
		summary_batch = []
		query_batch = []
		response_batch = []
		for documents in self.batcher.summary_set(documents_per_query):
			for summary,query,response in documents:
				# Turn documents into sequences of word indices and pad them
				summary = summary.split(" ")[:max_summary_words]
				summary = map(self.word_dict.word2int,summary)
				summary += [self.word_dict.word2int("<PAD>")]*(max_summary_words-len(summary))
				
				query = query.split(" ")[:max_query_words]
				query = map(self.word_dict.word2int,query)
				query += [self.word_dict.word2int("<PAD>")]*(max_query_words-len(query))
				
				summary_batch.append(summary)
				query_batch.append(query)
				response_batch.append(response)
				if len(summary_batch) == batch_size:
					yield summary_batch,query_batch,response_batch
					summary_batch = []
					query_batch = []
					response_batch = []
		yield summary_batch,query_batch,response_batch

	def iter_summaries(self,batch_size,max_summary_words):
		summary_batch = []
		title_batch = []
		for title,summary in self.batcher.iter_summaries():
			summary = summary.split(" ")[:max_summary_words]
			summary = map(self.word_dict.word2int,summary)
			summary += [self.word_dict.word2int("<PAD>")]*(max_summary_words-len(summary))
			summary_batch.append(summary)
			title_batch.append(title)

			if len(summary_batch)==batch_size:
				yield title_batch,summary_batch
				summary_batch = []
				title_batch = []
		yield title_batch,summary_batch