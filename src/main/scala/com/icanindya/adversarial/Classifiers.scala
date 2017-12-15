package com.icanindya.adversarial

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.tree.DecisionTree
import com.icanindya.adversarial.kdd99.FinalDataset
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.Row

object Classifiers {

  def randomForest(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 6 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    predictionAndLabel

  }

  def naiveBayes(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    predictionAndLabel
  }

  def logisticRegression(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(trainingData)

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    predictionAndLabel
  }

  def gradientBoostedTrees(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {

    // Train a GradientBoostedTrees model.
    // The defaultParams for Regression use SquaredError by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 10 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = Math.round(model.predict(features)).toDouble
        (prediction, label)
    }

    predictionAndLabel

  }

  def svm(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(trainingData, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = if (model.predict(features) > 0) 1.0 else 0.0
        (prediction, label)
    }

    predictionAndLabel

  }

  def decisionTree(trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val predictionAndLabel = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    predictionAndLabel
  }

  def multilayeredPerceptron(sc: SparkContext, trainingData: RDD[LabeledPoint], testData: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val spark = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    val numFeatures = trainingData.first.features.size
    
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 2(classes)
    val layers = Array[Int](numFeatures, 5, 4, 2)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model = trainer.fit(spark.createDataFrame(trainingData))
    
    // compute accuracy on the test set
    val result = model.transform(spark.createDataFrame(testData))
    val predictionAndLabels = result.select("prediction", "label")

    predictionAndLabels.map{(r: Row) =>
      (r(0).asInstanceOf[Double], r(1).asInstanceOf[Double])
    }.rdd
  }

  def printMetrics(predictionAndLabels: RDD[(Double, Double)], classifier: String) {
    val positives = predictionAndLabels.filter(_._2 == 1.0)
    val negatives = predictionAndLabels.filter(_._2 == 0.0)

    val tp = positives.filter(x => x._1 == x._2).count.toDouble
    val fp = positives.filter(x => x._1 != x._2).count.toDouble
    val tn = negatives.filter(x => x._1 == x._2).count.toDouble
    val fn = negatives.filter(x => x._1 != x._2).count.toDouble
    
    println("positives: " + positives.count)
    println("negatives: " + negatives.count)
    println("tp: " + tp)
    println("fp: " + fp)
    println("tn: " + tn)
    println("fn: " + fn)

    println(classifier + " accuracy: " + (tp + tn) / (tp + tn + fp + fn))
    println(classifier + " recall: " + tp / (tp + fn))
    println(classifier + " precision: " + tp / (tp + fp))
  }

}