import random

with open("../wiki_cache.csv") as file:
	trainfile = open("../wikitrain.csv","w")
	testfile = open("../wikitest.csv","w")
	for i,line in enumerate(file):
		if i % 1000 == 0:
			print i
		if random.random() > 0.92:
			testfile.write(line)
		else:
			trainfile.write(line)
	testfile.close()
	trainfile.close()

