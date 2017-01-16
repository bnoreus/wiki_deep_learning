# This Python file uses the following encoding: utf-8
# Author: Bure Noreus

from pymongo import MongoClient
from bson import ObjectId
import random

# This class retrieves wiki documents from mongoDB and builds (summary,query)-pairs.
class DocumentBatcher:
	def __init__(self):
		client = MongoClient("mongodb://localhost:27017")
		self.document_table = client["wikipedia"]["documents"]

	# Retrieves a set of [size] document summaries and randomly choses a paragraph that is not a summary for one of the documents.
	# Each summary is then paired with that random paragraph and a binary response (Was this the document the paragraph originally came from?)
	def summary_set(self,size):
		documents = []
		for i,doc in enumerate(self.document_table.find().batch_size(400)):
			paragraphs = doc["text"].split("\n")
			if len(paragraphs) > 2:
				documents.append(paragraphs)
			if len(documents) == size:
				# Sample a document to retrieve a query paragraph
				sample_idx = random.randint(0,size-1)
				query_doc = documents[sample_idx]
				query_paragraphs = query_doc[1:] # Remove the summary. We won't sample from that.
				query_paragraph = query_paragraphs[random.randint(0,len(query_paragraphs)-1)]
				
				# Build triplets (summary,query,response)
				for j,d in enumerate(documents):
					response = 1.0 if j==sample_idx else 0.0
					documents[j] = (d[0],query_paragraph,response)
				
				yield documents
				documents = []

	# Creates a iterator over every document used in the training data. This is used when visualing the results.
	def iter_summaries(self):
		for doc in self.document_table.find().batch_size(400):
			paragraphs = doc["text"].split("\n")
			if len(paragraphs) > 2:
				summary = paragraphs[0]
				yield doc["title"],summary