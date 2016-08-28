package com.icanindya.adversarial

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA
import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.reflect.io.Path
import java.io.FileWriter

object AttributeInfo {

  val attributesFile = Constants.attrFile
  val attributesInfoFile = "E:/Data/KDD99/attInfo"
  val trainDataFile = Constants.trainFile
  val testDataFile = Constants.testFile
  
  val numFeatures = 41

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/hadoop241/");
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("adversarial").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.memory", "10g")
    val sc = new SparkContext(conf)

    val trainData = sc.textFile(trainDataFile);
    val testData = sc.textFile(testDataFile)
    val data = trainData.union(testData);

    val fw = new FileWriter(attributesInfoFile, false)
    val lines = Source.fromFile(attributesFile).getLines().toArray
    try {
      for (i <- 0 to numFeatures - 1) {
        val attName = lines(i).split(":")(0).trim()
        val attType = lines(i).split(":")(1).trim()

        var info = ""
        if (attType == "continuous") {
          val values = data.map(_.split(',')(i).toDouble).distinct()
          info = "%d : %s : con : [%f - %f]\n".format(i, attName, values.min(), values.max())
        } else {
          val values = data.map(_.split(',')(i)).distinct()
          info = "%d : %s : sym : [%s]\n".format(i, attName, values.collect().mkString(","))
        }

        fw.write(info)
      }

    } finally fw.close()
  }

}