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

object AttrChangeThreshold {

  val attributesFile = Constants.attrFile
  val numAttr = Constants.numAttr
  val attributesInfoFile = "E:/Data/KDD99/attInfo"
  val trainDataFile = Constants.trainFile
  val testDataFile = Constants.testFile

  //attributes: 1 + 3 + 66 + 11 + 37 = 118
  var attrChangeThresholds: Array[Double] = Array.fill[Double](118)(0.0)

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/hadoop241/");
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("adversarial").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.memory", "10g")
    val sc = new SparkContext(conf)
  }
  
  def getArray(): Array[Double] = {
    val lines = Source.fromFile(attributesInfoFile).getLines().toArray
    for (i <- 0 to numAttr - 1) {
      val attName = lines(i).split(":")(1).trim()
      val attType = lines(i).split(":")(2).trim()
      var domain = lines(i).split(":")(3).trim()

      var changeThreshold = 0.0
      if (attType == "con") {
        domain = domain.substring(1, domain.length() - 2);
        val extremeVals = domain.split("-").map(_.trim().toDouble)
        changeThreshold = (extremeVals(1) - extremeVals(0)) * 0.5

        if (i == 0) attrChangeThresholds(i) = changeThreshold
        else {
          attrChangeThresholds(i + 1 + 3 + 66 + 11 - 4) = changeThreshold
        }
      }
//      attrChangeThresholds(i) = 1.0
    }
    return attrChangeThresholds
  }

}