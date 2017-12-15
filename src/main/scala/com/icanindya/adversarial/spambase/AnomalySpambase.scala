package com.icanindya.adversarial.spambase

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.classification.NaiveBayes
import scala.reflect.io.Path
import com.icanindya.adversarial.Spark

object AnomalySpambase {
  val trainDataFile = "E:/Data/Spambase/"
  val testDataFile = ""
  
  val rawDataFile = "E:/Data/Spambase/spambase.data"
  val modelDir = "E:/Data/Adversarial/Spambase/model"
  val pcFile = "E:/Data/Adversarial/Spambase/pc"
  
  val NUM_FEATURES = 57 //each line has 58 elements: 57 features, 1 class label
  
  def main(args: Array[String]){
    val sc = Spark.getContext()
    
    val wholeRawData = sc.textFile(rawDataFile)
    val scaledData = scaleData(wholeRawData)
    
   
    
    // Split data into training (60%) and test (40%).
    val Array(training, test) =  processData(wholeRawData).randomSplit(Array(0.6, 0.4), 123456)

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    
    println("Naive Bayes accuracy: " + accuracy)

    
    val spamSamples = scaleData(wholeRawData).filter { x => x.label == 1.0 }
    val hamSamples = scaleData(wholeRawData).filter { x => x.label == 0.0 }
    
    val trainSamples = hamSamples.sample(false, 0.5, 123456)
    val testSamples = hamSamples.subtract(trainSamples).union(spamSamples)
    
    println("Spam count: %d".format(spamSamples.count()))
    println("Ham count: %d".format(hamSamples.count()))
    println("Train count: %d".format(trainSamples.count()))
    println("Test count: %d".format(testSamples.count()))
    
    kMeansModel(sc, trainSamples, testSamples, 100)
    
  }
  
  def kMeansModel(sc: SparkContext, trainSamples: RDD[LabeledPoint], testSamples: RDD[LabeledPoint], k: Int){
    val labeledPointTrainRdd = trainSamples
    labeledPointTrainRdd.cache()
    
    println(labeledPointTrainRdd.count())
    
    val numPc = 30
    val mat = new RowMatrix(labeledPointTrainRdd.map(_.features))
    val pc = mat.computePrincipalComponents(numPc)
    val projTrain = mat.multiply(pc).rows
    
    val numClusters = k
    val numIterations = 10
    val kmeans = new KMeans().setK(numClusters).setMaxIterations(numIterations)
    val model = kmeans.run(projTrain)
    
    Path(modelDir).deleteRecursively()
    if(!Path(modelDir).exists) model.save(sc, modelDir)
    sc.parallelize(Array(pc)).saveAsObjectFile(pcFile)
    
    val pca = new PCA(numPc).fit(labeledPointTrainRdd.map(_.features))
    
    val labeledPointTestRdd = testSamples
    val projTest = labeledPointTestRdd.map { p => p.copy(features = pca.transform(p.features)) }
    
    var TP = 0
    var FP = 0
    var TN = 0
    var FN = 0
    val sqDistThres = 0.01
    
    projTest.collect().foreach{ x =>
      val clusterIndex = model.predict(x.features)
      val sqDist = Vectors.sqdist(x.features, model.clusterCenters(clusterIndex))
      System.out.println("label: %f, cluster: %d, distance: %f".format(x.label, clusterIndex, sqDist))
      
      if(sqDist < sqDistThres){
        if(x.label == 0.0) TN += 1
        else FN += 1
      }
      else{
        if(x.label == 0.0) FP += 1
        else TP += 1
      }
    }
    println("TP is : " + TP)
    println("FP is : " + FP)
    println("TN is : " + TN)
    println("FN is : " + FN)
    println("precision is : " + TP / (TP + FP).toFloat)
    println("recall is : " + TP / (TP + FN).toFloat)
    println("Accuracy is :" + (TP + TN) / (TP + TN + FP + FN).toFloat)
  }
  
  def processData(data: RDD[String]): RDD[LabeledPoint] = {
    data.map{ line =>
      val values = line.split(",", -1).map(_.toDouble).toBuffer
      val label = values.remove(values.length - 1)
      new LabeledPoint(label, Vectors.dense(values.toArray))
    }
  }
  
  
   def scaleData(data : RDD[String]): RDD[LabeledPoint] = {
     
     val featureMinMax = (0 to NUM_FEATURES - 1).map{ i =>
       val vals = data.map{ line => 
         line.split(",", -1)(i).toDouble
       }
       (i, (vals.min(), vals.max()))
     }.toMap
     
     println(featureMinMax)
     
     data.map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      val label = values(NUM_FEATURES)
//      println(label)
      
      val scaledFeatures = for (i <- 0 to NUM_FEATURES - 1) yield {
        if(values(i) < featureMinMax(i)._1) values(i) = featureMinMax(i)._1
        else if(values(i) > featureMinMax(i)._2) values(i) = featureMinMax(i)._2
        (values(i) - featureMinMax(i)._1)/(featureMinMax(i)._2 - featureMinMax(i)._1)
      }

      new LabeledPoint(label, Vectors.dense(scaledFeatures.toArray))
     }
   }
  
}