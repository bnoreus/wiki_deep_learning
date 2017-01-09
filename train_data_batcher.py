# This Python file uses the following encoding: utf-8
# Author: Bure Noreus
from document_batcher import DocumentBatcher
from word_dict import WordDict

class TrainDataBatcher:
	def __init__(self,word_dict):
		self.batcher = DocumentBatcher()
		self.word_dict = word_dict
	# Creates a cache of the original MongoDB. The cache is very memory efficient because it stores word indices, not words. 
	# In order to recover the words, the original WordDict object is needed.
	def create_cache(self):
		with open("wiki_cache.csv","w") as file:
			for batch_idx,(summary_batch,query_batch,response_batch) in enumerate(self.mini_batch(500,2,100,100)):
				if batch_idx % 1000 == 0:
					print "Saved ",batch_idx*500, " rows of data"
				for i in range(500):
					summary = summary_batch[i]
					query = query_batch[i]
					response = response_batch[i]
					file.write(" ".join(map(str,summary))+","+" ".join(map(str,query))+","+str(response)+"\n")

	# Retreive a data batch from a data cache
	# This assumes that you have created a cache CSV file with the function create_cache() described above.
	# I also recommend create a cache file and splitting that file randomly into a test a train set.
	def mini_batch_from_cache(self,batch_size,file_name):
		summary_batch = []
		query_batch = []
		response_batch = []
		for line in open(file_name):
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