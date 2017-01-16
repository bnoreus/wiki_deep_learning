# Deep Learning on the English Wikipedia Dataset

The goal of this project is to create a Question Answering bot. To get there, I have to solve various Information Retrieval tasks.
In this task, I am trying to use convolutional neural networks to encode wikipedia article summaries. 


# Creating Data
The Wikipedia dump is downloaded as "enwiki-20161201-pages-articles.xml.bz2". This archive is then unzipped and parsed with
a python script. (http://medialab.di.unipi.it/wiki/Wikipedia_Extractor , https://github.com/attardi/wikiextractor).

The parser will create lots of files, each containg ~5000 lines. Before performing any machine learning, these need a little
pre-processing. This is done with the Apache Spark application found in the folder "TextCleaner"

To run this Spark app, you need to have spark installed. The app can be run with the "run_spark.sh" script.
To get the script to run, you need to edit the script such that it points to the path at which your Apache Spark installation is.

