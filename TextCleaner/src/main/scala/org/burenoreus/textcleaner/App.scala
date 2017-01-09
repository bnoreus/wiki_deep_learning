package org.burenoreus.textcleaner
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import java.io.File
import scala.io.Source


import com.mongodb.BasicDBObject;
import org.bson.BSONObject
import com.mongodb.hadoop.MongoOutputFormat
import org.apache.hadoop.conf.Configuration
import org.bson.types.ObjectId;
/*
The purpose of this app is to clean the text of a wikipedia dump so that the text can be used for deep learning in tensorflow.
Disclaimer: The raw wikipedia-XML has already been parsed with the follownig tool: https://github.com/attardi/wikiextractor

*/

object App {
	val sc = new SparkContext(new SparkConf());
	def main(args: Array[String]) = {
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		println("In which folder are the wikipedia files located?")
		val wikiFolder = "../raw_data"
		val files = findTextFiles(wikiFolder)
		val urls = files.map{f => f.getPath()}
		val documents = documentRDD(urls).coalesce(1000)
		saveToMongoDB(cleanDocuments(documents))
	}
	// Take an RDD of documents and clean the text such that it is more suited for ML
	def cleanDocuments(documents : RDD[String]) : RDD[String] = {
		documents.map{document => 
			document
			.replace("\"\"","\"")
			.replace("\""," \" ")
			.replace("\' "," ")
			.replace(" \'"," ")
			.replace("/"," ")
			.replace("(","( ")
			.replace(")"," )")
			.replace(". "," . ")
			.replace(", "," , ")
			.replace("?"," ?")
			.replace(":"," :")
			.replace(";"," ;")
			.replace("<BR>","")
			.replaceAll("[ ]+"," ")
			.toLowerCase
		}
	}
	// Get an RDD whose rows are Wiki articles as Strings
	def documentRDD(urls : Array[String]) : RDD[String] = {
		sc.parallelize(urls)
		.map{url => 
			var documents = scala.collection.mutable.ArrayBuffer[String]();
			var document = ""
			for(line <- Source.fromFile(url).getLines) {
				if(line.slice(0,4)=="<doc"){
					document = ""
				} else if(line == "</doc>"){
					documents += document
				} else {
					document += line+"\n"
				}
			}
			documents
		}.flatMap{i=>i}
	}

	// Find all files in a given directory
	def findTextFiles(folder : String) = {
		// Recursive function that starts at some folder/file and exhaustively searches for every file
		def recursiveSearch(d: File): Array[File] = {
		    val (dirs, files) =  d.listFiles.partition(_.isDirectory)
		    files ++ dirs.flatMap(recursiveSearch)
		}
		val startingFolder = new File(folder)
		recursiveSearch(startingFolder)
	}
	def saveToMongoDB(documents : RDD[String]) : Unit = {
		val outputConfig = new Configuration()
		outputConfig.set("mongo.output.uri","mongodb://127.0.0.1:27017/wikipedia.documents");

		val bsonDocs : RDD[(ObjectId,BSONObject)] = documents.map{d => 
			val (title,text) = {
				val lines = d.split("\n")
				(lines(0),lines.slice(2,lines.length).mkString("\n"))
			}
			val doc = new BasicDBObject();
			doc.put("title",title)
			doc.put("text",text)
			(new ObjectId(),doc)
		}
		bsonDocs.saveAsNewAPIHadoopFile(
			"not-used.com",
			classOf[Object],
			classOf[BSONObject],
			classOf[MongoOutputFormat[Object,BSONObject]],
			outputConfig
		)
	}
}
