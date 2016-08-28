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


object Approach2 {
  
  val trainDataFile = "E:/Data/kddcupdata/kddcup.trasfrom.normal"
  val testDataFile = "E:/Data/KDD99/testing"
  val modelDir = "E:/Data/KDD99/Metadata/Approach2/Model"
  
  var protocols: Map[String, Int] =  Map()
  var services: Map[String, Int] =  Map()
  var flags: Map[String, Int] =  Map()
  var threshold = 75000
  var TP = 0L
  var FP = 0L
  var FN = 0L
  var TN = 0L
  
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

    protocols = data.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    services = data.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    flags = data.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

    val labeledPointTrainRdd = processData(trainData)

    val numPc = 30
    val pca = new PCA(numPc).fit(labeledPointTrainRdd.map(_.features))
    val projTrain = labeledPointTrainRdd.map(p => p.copy(features = pca.transform(p.features)))
    
    val numClusters = 100
    val numIterations = 10
    val kmeans = new KMeans().setK(numClusters).setMaxIterations(numIterations)
    val model = kmeans.run(projTrain.map(_.features))
    
    Path(modelDir).deleteRecursively()
    if(!Path(modelDir).exists) model.save(sc, modelDir)
    
    val labeledPointTestRdd = processData(testData)
    val projTest = labeledPointTestRdd.map { p => p.copy(features = pca.transform(p.features)) }
    
    projTest.collect().foreach{ x =>
      val clusterIndex = model.predict(x.features)
      if(Vectors.sqdist(x.features, model.clusterCenters(clusterIndex)) < threshold){
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
  
  def processData(data : RDD[String]): RDD[LabeledPoint] = {
    data.map { line =>
      
      val buffer = line.split(",").toBuffer
      val protocol = buffer.remove(1)
      val service = buffer.remove(1)
      val tcpState = buffer.remove(1)
      val label = if(buffer.remove(buffer.length - 1) == "normal.") 0.0 else 1.0
      val vector = buffer.map(_.toDouble)

      val newProtocolFeatures = new Array[Double](protocols.size)
      newProtocolFeatures(protocols(protocol)) = 1.0
      val newServiceFeatures = new Array[Double](services.size)
      newServiceFeatures(services(service)) = 1.0
      val newTcpStateFeatures = new Array[Double](flags.size)
      newTcpStateFeatures(flags(tcpState)) = 1.0

      vector.insertAll(1, newTcpStateFeatures)
      vector.insertAll(1, newServiceFeatures)
      vector.insertAll(1, newProtocolFeatures)

      new LabeledPoint(label, Vectors.dense(vector.toArray))
    }
  }
  
}
