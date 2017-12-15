package com.icanindya.adversarial.iscx

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import com.icanindya.adversarial.Classifiers
import com.icanindya.adversarial.Spark
import org.apache.spark.rdd.RDD

object Classify {

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    
    
    
//   val data = sc.textFile(FinalDataset.joinedDsFile.format(11))
//    .union(sc.textFile(FinalDataset.joinedDsFile.format(12)))
//    .union(sc.textFile(FinalDataset.joinedDsFile.format(13)))
//    .union(sc.textFile(FinalDataset.joinedDsFile.format(14)))
//    .union(sc.textFile(FinalDataset.joinedDsFile.format(15)))
//    .union(sc.textFile(FinalDataset.joinedDsFile.format(16)))
//    .union(sc.textFile(FinalDataset.joinedDsFile.format(17)))
//    
//   println("Attack count: " + data.filter(_.endsWith("Attack")).count)
//   println("Attack count: " + data.filter(_.endsWith("Attack")).distinct.count) 
    
    val trainingData = sc.textFile(FinalDataset.finalTrainFile)
      .map { x =>
        val tokens = x.split(",", -1)
        val label = tokens.last.toDouble
        val features = tokens.dropRight(1).map(_.toDouble)
        new LabeledPoint(label, Vectors.dense(features))
      }

    val testData = sc.textFile(FinalDataset.finalTestFile)
      .map { x =>
        val tokens = x.split(",", -1)
        val label = tokens.last.toDouble
        val features = tokens.dropRight(1).map(_.toDouble)
        new LabeledPoint(label, Vectors.dense(features))
      }
    
//    println(trainingData.union(testData).filter { x => x.label == 1.0 }.count())
//
    var predictionAndLabels: RDD[(Double, Double)] = null
    
    
    
    predictionAndLabels = Classifiers.randomForest(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Random Forest")
    predictionAndLabels = Classifiers.naiveBayes(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Naive Bayes")
    predictionAndLabels = Classifiers.logisticRegression(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Logistic Regression")
    predictionAndLabels = Classifiers.gradientBoostedTrees(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Gradient Boosted Trees")
    predictionAndLabels = Classifiers.svm(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Support Vector Machine")
    predictionAndLabels = Classifiers.multilayeredPerceptron(sc, trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Multilayer Perceptron")

  }

}