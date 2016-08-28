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
import org.apache.spark.mllib.clustering.KMeansModel
import java.io.FileWriter
import java.io.PrintWriter

object Attack {

  val trainDataFile = Constants.trainFile
  val testDataFile = Constants.testFile
  val modelDir = "E:/Data/KDD99/Metadata/Approach2New/Model"
  val pcFile = "E:/Data/KDD99/Metadata/Approach2New/pc"
  val pcPrintFile = "E:/Data/KDD99/Metadata/Approach2New/pc_print"
  
  var protocols: Map[String, Int] = Map()
  var services: Map[String, Int] = Map()
  var flags: Map[String, Int] = Map()
  var sqDistThres = Constants.sqDistThres
  var TP = 0L
  var FP = 0L
  var FN = 0L
  var TN = 0L
  var succ = 0L;


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
    val data = trainData.union(testData)

    protocols = data.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    services = data.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    flags = data.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

    val labeledPointTrainRdd = processData(trainData)

    val model = KMeansModel.load(sc, modelDir)
    val pc = sc.objectFile[Matrix](pcFile).collect()(0)
    
    val pcArr = pcToArr(pc)
    printPc(pcArr, pcPrintFile)

    val projCentersArr = model.clusterCenters.map(_.toArray)
    println("Feature change thresholds: %s".format(AttrChangeThreshold.getArray().mkString(",")))
    val optimizer = new Optimizer(pcArr, projCentersArr, Math.sqrt(sqDistThres), AttrChangeThreshold.getArray())
    
    var numAssocTestPoints = Array.fill[Long](projCentersArr.length)(0)
    var totSqDist = Array.fill[Double](projCentersArr.length)(0)
    var minSqDist = Array.fill[Double](projCentersArr.length)(Double.MaxValue)
    var maxSqDist = Array.fill[Double](projCentersArr.length)(Double.MinValue)
    
    val labeledPointTestRdd = processData(testData)

    labeledPointTestRdd.collect().foreach {  lp =>
          var pointMat = new DenseMatrix(1, lp.features.size, lp.features.toArray)
          var projPoint = pointMat.multiply(new DenseMatrix(pc.numRows, pc.numCols, pc.toArray));
          var projPointVec = new DenseVector(projPoint.toArray);
          
          val clusterIndex = model.predict(projPointVec)
          val sqDist = Vectors.sqdist(projPointVec, model.clusterCenters(clusterIndex))
          
          numAssocTestPoints(clusterIndex) += 1
          totSqDist(clusterIndex) += sqDist
          if(sqDist < minSqDist(clusterIndex)) minSqDist(clusterIndex) = sqDist
          if(sqDist > maxSqDist(clusterIndex)) maxSqDist(clusterIndex) = sqDist
          
          
          if(sqDist < sqDistThres){
            if(lp.label == 0.0) TN += 1
            else FN += 1
          }
          else{
            if(lp.label == 1.0){
               println("Cluster %d, Squared distance: %f".format(clusterIndex, sqDist))
              TP += 1
              optimizer.setSqDist(sqDist);
              if(optimizer.optimize(lp.features.toArray, projPointVec.toArray)) succ += 1
            }
            else FP += 1
          }
        }

    println("Successful mimicry attacks: " + succ)
    println("TP is : " + TP)
    println("FP is : " + FP)
    println("TN is : " + TN)
    println("FN is : " + FN)
    println("precision is : " + TP / (TP + FP).toFloat)
    println("recall is : " + TP / (TP + FN).toFloat)
    println("Accuracy is :" + (TP + TN) / (TP + TN + FP + FN).toFloat)
    
//    for(i <- 0 to projCentersArr.size - 1){
//      println("\nCluster %d:".format(i))
//      println("Num of associated test points: %d".format(numAssocTestPoints(i)))
//      println("Avg squared distance of associated points: %f".format(totSqDist(i)/numAssocTestPoints(i)))
//      println("Min squared distance: %f".format(minSqDist(i)))
//      println("Max squared distance: %f".format(maxSqDist(i)))
//    }
    
    sc.stop()
  }

  def processData(data: RDD[String]): RDD[LabeledPoint] = {
    data.map { line =>

      val buffer = line.split(",").toBuffer
      val protocol = buffer.remove(1)
      val service = buffer.remove(1)
      val tcpState = buffer.remove(1)
      val label = if (buffer.remove(buffer.length - 1) == "normal.") 0.0 else 1.0
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

  def pcToArr(pc: Matrix): Array[Array[Double]] = {
    var c = 0
    val pcArr = Array.ofDim[Double](pc.numRows, pc.numCols)

    //pc.toArray is a column major array
    for (i <- 0 to pc.numCols - 1) {
      for (j <- 0 to pc.numRows - 1) {
        pcArr(j)(i) = pc.toArray(c)
        c += 1
      }
    }

    pcArr
  }
  
  def printPc(pcArr: Array[Array[Double]], outFile: String){
    val fw = new FileWriter(pcPrintFile)
    val builder = new StringBuilder()
    try{
      for(i <- 0 to pcArr.length - 1){
        fw.write(pcArr(i).mkString(","))
        fw.write("\n")
      }
    }
    finally{
      fw.close()
    }
  }
  
  
}
