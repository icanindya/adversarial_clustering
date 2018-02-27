package com.icanindya.adversarial.pdf

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.commons.math3.geometry.Space
import com.icanindya.adversarial.Spark
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import java.io.File
import java.io.PrintWriter

object FinalDataset {

  val pdfMaliciousFile = "D:/Data/PDF/original/malicious_original_features.txt"
  val pdfBenignFile = "D:/Data/PDF/original/benign_original_features.txt"
  val pdfAllOriginalFile = "D:/Data/PDF/original/all_original_features.txt"

  val finalTrainFile = "D:/Data/PDF/final/train"
  val finalTestFile = "D:/Data/PDF/final/test"
  
  val randomTestSamplesFile = "D:/Data/PDF/final/random_test_%d"

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    saveFinalDatasets(sc)
    getFinalDatasetStat(sc, finalTrainFile, finalTestFile)
  }
  
  def saveFinalDatasets(sc: SparkContext){
    
    val data = makeLabeledOriginalDataset(sc.textFile(pdfBenignFile), sc.textFile(pdfMaliciousFile)).distinct()
    
    val fixedCatData = fixCategoricalFeatures(data)
    fixedCatData.cache()
    
    val scaledData = scaleData(fixedCatData)
    val shrinkedData = scaledData.map{x =>
       x.split(",", -1).map(v => "%.5f".format(v.toDouble)).mkString(",")
    }
    .distinct
    shrinkedData.cache() 

    val trainData = shrinkedData.sample(false, 0.5, 123456)
    trainData.cache()
    
    val testData = shrinkedData.subtract(trainData)
    testData.cache()
    
    trainData.coalesce(1).saveAsTextFile(finalTrainFile)
    testData.coalesce(1).saveAsTextFile(finalTestFile)
  }
  
  def makeLabeledOriginalDataset(benignData: RDD[String], maliciousData: RDD[String]): RDD[String] = {
    
//    val pw = new PrintWriter(new File(pdfAllOriginalFile))
    
    val data = benignData.map(line => line.concat(",0"))
    .union(maliciousData.map(line => line.concat(",1")))
  
//    data.collect.foreach { line =>
//      pw.println(line)
//    }
//    
//    pw.close
    
    data
        
  }

  def fixCategoricalFeatures(data: RDD[String]): RDD[String] = {

    data.map { line =>
      val features = line.split(",")
      if(features(8) == "True") features(8) = "1"
      else features(8) = "0"
      
      if(features(92) == "True") features(92) = "1"
      else features(92) = "0"
      
      features.mkString(",")
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

      val label = if (tokens(NUM_FEATURES) == "0") "0.0" else "1.0"
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
    val trainBenignCount = trainLabels.filter { _ == 0.0 }.count
    val trainMaliciousCount = trainLabels.filter { _ == 1.0 }.count
    
    val testLabels = sc.textFile(testFile).map(_.split(",", -1).map(_.toDouble).last)
    testLabels.cache
    val testBenignCount = testLabels.filter { _ == 0.0 }.count
    val testMaliciousCount = testLabels.filter { _ == 1.0 }.count
    
    println(trainBenignCount, trainMaliciousCount)
    println(testBenignCount, testMaliciousCount)
    
    
  }
  
  def saveRandomTestSamples(sc: SparkContext, sampleSize: Int){
    val testRdd = sc.textFile(finalTestFile)
    testRdd.cache
    sc.parallelize(testRdd.takeSample(false, sampleSize, 123456)).coalesce(1).saveAsTextFile(randomTestSamplesFile.format(sampleSize))
    
  }

}