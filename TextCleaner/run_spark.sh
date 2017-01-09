#!/bin/sh


# This is my spark installation path. Change this to wherever yours is installed
SPARK_INSTALLATION_PATH=~/spark-1.6.1-bin-hadoop2.6/bin/spark-submit



$SPARK_INSTALLATION_PATH \
--class org.burenoreus.textcleaner.App \
--master local[4] \
--executor-memory 7300m \
--driver-memory 7300m \
--jars dependencies/mongo-hadoop-core-1.4.0.jar,dependencies/mongo-java-driver-3.2.0.jar \
target/textcleaner-0.0.1-SNAPSHOT.jar
