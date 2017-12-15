package com.icanindya.adversarial.iscx

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import com.icanindya.adversarial.Spark

object FinalDataset {
  val joinedDsFile = "D:/Data/ISCX IDS/joined/%d" //.format(int)

  val finalTrainFile = "D:/Data/ISCX IDS/final/train"
  val finalTestFile = "D:/Data/ISCX IDS/final/test"

  def main(args: Array[String]): Unit = {
    val sc = Spark.getContext()
    getFinalDatasetStat(sc)
  }

  def saveFinalDatasets(sc: SparkContext): Unit = {
    val data = sc.textFile(joinedDsFile.format(11))
      .union(sc.textFile(joinedDsFile.format(12)))
      .union(sc.textFile(joinedDsFile.format(13)))
      .union(sc.textFile(joinedDsFile.format(14)))
      .union(sc.textFile(joinedDsFile.format(15)))
      .union(sc.textFile(joinedDsFile.format(16)))
      .union(sc.textFile(joinedDsFile.format(17)))

    val scaledData = scaleData(data)
    val shrinkedData = scaledData.map { x =>
      x.split(",", -1).map(v => "%.5f".format(v.toDouble)).mkString(",")
    }
    .distinct
    shrinkedData.cache()
    
    println(shrinkedData.filter { _.endsWith("1.00000")}.count)
    


    val trainData = shrinkedData.sample(false, 0.5, 123456)
    trainData.cache()
    
    println(trainData.filter { _.endsWith("1.00000")}.count)
    

    val testData = shrinkedData.subtract(trainData)
    testData.cache()
    
    println(testData.filter { _.endsWith("1.00000")}.count)
    
    trainData.coalesce(1).saveAsTextFile(finalTrainFile)
    testData.coalesce(1).saveAsTextFile(finalTestFile)
  }

  def scaleData(data: RDD[String]): RDD[String] = {

    val NUM_FEATURES = data.first().split(",", -1).size - 1

    val featureMinMax = (0 to NUM_FEATURES - 1).map { i =>
      val vals = data.map { line =>
        line.split(",", -1)(i).toDouble
      }
      (i, (vals.min(), vals.max()))
    }.toMap

    val scaledLine = data.map { line =>
      val tokens = line.split(",", -1)
      val label = if (tokens(NUM_FEATURES) == "Normal") "0.0" else "1.0"
      val values = tokens.dropRight(1).map(_.toDouble)

      val scaledFeatures = for (i <- 0 to NUM_FEATURES - 1) yield {
        if (values(i) < featureMinMax(i)._1) values(i) = featureMinMax(i)._1
        else if (values(i) > featureMinMax(i)._2) values(i) = featureMinMax(i)._2

        var scaledVal = 0.0
        if (featureMinMax(i)._2 - featureMinMax(i)._1 != 0.0) scaledVal = (values(i) - featureMinMax(i)._1) / (featureMinMax(i)._2 - featureMinMax(i)._1)

        scaledVal
      }
      scaledFeatures.mkString(",") + "," + label
    }
    scaledLine
  }
  
  def getFinalDatasetStat(sc: SparkContext){
    val trainLabels = sc.textFile(finalTrainFile).map(_.split(",", -1).map(_.toDouble).last)
    trainLabels.cache
    val trainBenignCount = trainLabels.filter { _ == 0.0 }.count
    val trainMaliciousCount = trainLabels.filter { _ == 1.0 }.count
    
    val testLabels = sc.textFile(finalTestFile).map(_.split(",", -1).map(_.toDouble).last)
    testLabels.cache
    val testBenignCount = testLabels.filter { _ == 0.0 }.count
    val testMaliciousCount = testLabels.filter { _ == 1.0 }.count
    
    println(trainBenignCount, trainMaliciousCount)
    println(testBenignCount, testMaliciousCount)
    
    
  }
}