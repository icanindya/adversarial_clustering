package com.icanindya.adversarial.kdd99

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.commons.math3.geometry.Space
import com.icanindya.adversarial.Spark
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import com.icanindya.adversarial.AdvUtil

object FinalDataset {

  val kddTrainFile = "D:/Data/KDD99/full/kddcup.data.corrected"
  val kddTestFile = "D:/Data/KDD99/test/corrected"

  val finalTrainFile = "D:/Data/KDD99/final/train"
  val finalTestFile = "D:/Data/KDD99/final/test"
  val finalValidationFile = "D:/Data/KDD99/final/validation"
  
  val randomTestSamplesFile = "D:/Data/KDD99/final/random_test_%d"
  
  val trainSize = 350000
  val validationEachSize = 10000

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    saveFinalDatasets(sc)
  }
  
  def saveFinalDatasets(sc: SparkContext){
    val data = sc.textFile(kddTrainFile).union(sc.textFile(kddTestFile)).distinct()

    val fixedCatData = fixCategoricalFeatures(data)
    fixedCatData.cache()
    
    val scaledData = scaleData(fixedCatData)
    val shrinkedData = scaledData.map{x =>
       x.split(",", -1).map(v => "%.5f".format(v.toDouble)).mkString(",")
    }
    .distinct
    shrinkedData.cache()
    
    val benValidationData = sc.parallelize(shrinkedData.filter(_.endsWith("0.00000")).takeSample(false, validationEachSize, 123456))
    val malValidationData = sc.parallelize(shrinkedData.filter(_.endsWith("1.00000")).takeSample(false, validationEachSize, 123456))
    val validationData = benValidationData.union(malValidationData)
    
    val trainAndTestData = shrinkedData.subtract(validationData)
    trainAndTestData.cache()
    
    val benTrainAndTestData = trainAndTestData.filter(_.endsWith("0.00000"))
    benTrainAndTestData.cache()
    
    val trainData = benTrainAndTestData.sample(false, trainSize.toDouble/benTrainAndTestData.count, 123456)    
    val testData = trainAndTestData.subtract(trainData)

    trainData.coalesce(1).saveAsTextFile(finalTrainFile)
    validationData.coalesce(1).saveAsTextFile(finalValidationFile)   
    testData.coalesce(1).saveAsTextFile(finalTestFile)
   
  }

  def fixCategoricalFeatures(data: RDD[String]): RDD[String] = {
    val protocols = data.map(_.split(",", -1)(1)).distinct().collect().zipWithIndex.toMap
    val services = data.map(_.split(",", -1)(2)).distinct().collect().zipWithIndex.toMap
    val flags = data.map(_.split(",", -1)(3)).distinct().collect().zipWithIndex.toMap
    
    println("protocols size: " + protocols.size)
    println("services size: " + services.size)
    println("flags size: "+ flags.size)
    

    data.map { line =>
      val buffer = line.split(",").toBuffer
      val protocol = buffer.remove(1)
      val service = buffer.remove(1)
      val tcpState = buffer.remove(1)

      val newProtocolFeatures = Array.fill[String](protocols.size)("0.0")
      newProtocolFeatures(protocols(protocol)) = "1.0"
      val newServiceFeatures = Array.fill[String](services.size)("0.0")
      newServiceFeatures(services(service)) = "1.0"
      val newTcpStateFeatures = Array.fill[String](flags.size)("0.0")
      newTcpStateFeatures(flags(tcpState)) = "1.0"

      buffer.insertAll(1, newTcpStateFeatures)
      buffer.insertAll(1, newServiceFeatures)
      buffer.insertAll(1, newProtocolFeatures)

      buffer.mkString(",")
    }
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

      val label = if (tokens(NUM_FEATURES) == "normal.") "0.0" else "1.0"
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
  
  def getFinalDatasetStat(sc: SparkContext, trainFile: String, testFile: String){
    val trainLabels = sc.textFile(trainFile).map(_.split(",", -1).map(_.toDouble).last)
    trainLabels.cache
    val benTrainCount = trainLabels.filter { _ == 0.0 }.count
    val malTrainCount = trainLabels.filter { _ == 1.0 }.count
    
    val validationLabels = sc.textFile(finalValidationFile).map(_.split(",", -1).map(_.toDouble).last)
    validationLabels.cache
    val benValidationCount = validationLabels.filter { _ == 0.0 }.count
    val malValidationCount = validationLabels.filter { _ == 1.0 }.count
    
    val testLabels = sc.textFile(testFile).map(_.split(",", -1).map(_.toDouble).last)
    testLabels.cache
    val benTestCount = testLabels.filter { _ == 0.0 }.count
    val malTestCount = testLabels.filter { _ == 1.0 }.count
    
    println("train: ben = %d, mal = %d".format(benTrainCount, malTrainCount))
    println("validation: ben = %d, mal = %d".format(benValidationCount, malValidationCount))
    println("test: ben = %d, mal = %d".format(benTestCount, malTestCount))
  }
  
  def saveRandomTestSamples(sc: SparkContext, sampleSize: Int){
    val testRdd = sc.textFile(finalTestFile)
    testRdd.cache
    sc.parallelize(testRdd.takeSample(false, sampleSize, 123456)).coalesce(1).saveAsTextFile(randomTestSamplesFile.format(sampleSize))
    
  }

}