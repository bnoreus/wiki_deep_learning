package com.eqt.app
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD


object App {
	val sc = new SparkContext(new SparkConf());
	def main(args: Array[String]) = {
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		println("Hello, Öhman")
	}

}
