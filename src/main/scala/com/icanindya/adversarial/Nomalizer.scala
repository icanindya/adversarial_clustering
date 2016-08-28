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

import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.reflect.io.Path
import scala.io.Source

object Nomalizer {
  val trainDataFile = "E:/Data/kddcupdata/kddcup.trasfrom.normal"
  val testDataFile = "E:/Data/KDD99/testing"
  val attributesFile = "E:/Data/KDD99/attributes"
  val stdTrainDataFile = "E:/Data/KDD99/std_ben_train"
  val stdTestDataFile = "E:/Data/KDD99/std_test"
  val scldTrainDataFile = "E:/Data/KDD99/scld_ben_train"
  val scldTestDataFile = "E:/Data/KDD99/scld_test"
  
  val STD = 0
  val SCL = 1

  var protocols: Map[String, Int] = Map()
  var services: Map[String, Int] = Map()
  var flags: Map[String, Int] = Map()

  val numAttr = 41

  val means: Array[Double] = Array.ofDim[Double](numAttr)
  val stDevs: Array[Double] = Array.ofDim[Double](numAttr)
  val mins: Array[Double] = Array.ofDim[Double](numAttr)
  val maxs: Array[Double] = Array.ofDim[Double](numAttr)
  val conAttr: Array[Int] = Array.fill[Int](numAttr)(0)

  def main(args: Array[String]) {
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

    val lines = Source.fromFile(attributesFile).getLines().toArray
    
//    print("arr size: " + stDevs.size)

    for (i <- 0 to numAttr - 1) {
      val attName = lines(i).split(":")(0).trim()
      val attType = lines(i).split(":")(1).trim()

      var info = ""
      if (attType == "continuous") {
        conAttr(i) = 1
        means(i) = data.map(_.split(',')(i).toDouble).mean()
        stDevs(i) = data.map(_.split(',')(i).toDouble).stdev()
        mins(i) = data.map(_.split(',')(i).toDouble).min()
        maxs(i) = data.map(_.split(',')(i).toDouble).max()
//        println("%d. %s: mean: %f std_dev: %f min: %f max: %f".format(i, attName, means(i), stDevs(i), mins(i), maxs(i)))
      }
    }
    trainData.map(normalize(_, STD)).coalesce(1).saveAsTextFile(stdTrainDataFile)
    trainData.map(normalize(_, SCL)).coalesce(1).saveAsTextFile(scldTrainDataFile)
    testData.map(normalize(_, STD)).coalesce(1).saveAsTextFile(stdTestDataFile)
    testData.map(normalize(_, SCL)).coalesce(1).saveAsTextFile(scldTestDataFile)
  }

  def normalize(line: String, mode: Int): String = {
    val values = line.split(",").map(_.trim())
    for (i <- 0 to numAttr - 1) {
      if (conAttr(i) == 1) {
        val value = values(i).toDouble
        if(maxs(i) - mins(i) == 0) values(i) = 0.0.toString()
        else if(mode == STD) values(i) = "%.5f".format(((value - means(i)) / stDevs(i)))
        else if(mode == SCL) values(i) = "%.5f".format(((value - mins(i)) / (maxs(i) - mins(i))))
      }
    }
    values.mkString(",")
  }
}