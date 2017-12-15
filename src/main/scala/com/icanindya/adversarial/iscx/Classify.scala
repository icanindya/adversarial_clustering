package com.icanindya.adversarial.iscx

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import com.icanindya.adversarial.Classifiers
import com.icanindya.adversarial.Spark
import org.apache.spark.rdd.RDD

object Classify {

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    
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

    var predictionAndLabels: RDD[(Double, Double)] = null
    
    predictionAndLabels = Classifiers.randomForest(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Random Forest")
    predictionAndLabels = Classifiers.decisionTree(trainingData, testData)
    Classifiers.printMetrics(predictionAndLabels, "Decision Tree")
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