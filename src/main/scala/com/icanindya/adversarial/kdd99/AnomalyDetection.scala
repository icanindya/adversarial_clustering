package com.icanindya.adversarial.kdd99

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
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.clustering.BisectingKMeans
import shapeless.ops.nat.ToInt
import java.io.PrintWriter
import java.io.File
import org.apache.spark.mllib.clustering.BisectingKMeans
import scala.collection.mutable.ListBuffer
import com.icanindya.adversarial.ClusterInfo
import com.icanindya.adversarial.AdvUtil
import org.apache.spark.mllib.clustering.BisectingKMeansModel
import scala.io.Source
import com.icanindya.adversarial.CplexOptimizer
import com.icanindya.adversarial.CplexSolution

object AnomalyDetection {

  val kmModelDir = "D:/Data/KDD99/model/kmeans/%d"
  val kmModelClusterFile = kmModelDir + "/clsuters.txt"
  val bkmModelDir = "D:/Data/KDD99/model/bkmeans/%d"
  val bkmModelClusterFile = bkmModelDir + "/clsuters.txt"
  val gmModelDir = "D:/Data/KDD99/model/gm/%d"
  val gmModelClusterFile = gmModelDir + "/clsuters.txt"

  val pcFile = "D:/Data/KDD99/model/pc/%d"
  
  val kmTpPointsFile = "D:/Data/KDD99/model/kmeans/%d/tp_points.txt"
  val kmTpClustersFile = "D:/Data/KDD99/model/kmeans/%d/tp_clusters.txt"
  val bkmTpPointsFile = "D:/Data/KDD99/model/bkmeans/%d/tp_points.txt"
  val bkmTpClustersFile = "D:/Data/KDD99/model/bkmeans/%d/tp_clusters.txt"
  val gmTpPointsFile = "D:/Data/KDD99/model/gm/%d/tp_points.txt"
  val gmTpClustersFile = "D:/Data/KDD99/model/gm/%d/tp_clusters.txt"
  
  val chebyshevK = 3

  def main(args: Array[String]) {
    val sc = Spark.getContext()

    val labeledPointTrainRdd = sc.textFile(FinalDataset.finalTrainFile).map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }
      .filter(_.label == 0.0)

    labeledPointTrainRdd.cache

    val labeledPointTestRdd = sc.textFile(FinalDataset.finalTestFile).map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }

    labeledPointTestRdd.cache
    
    gmAnomalyDetectionResult(sc, labeledPointTestRdd, 24, 30)

  }

  def pcVariaance(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint]) {
    val numFeatures = labeledPointTrainRdd.take(1)(0).features.size
    var mat = new RowMatrix(labeledPointTrainRdd.map(_.features))

    val variances = mat.computePrincipalComponentsAndExplainedVariance(numFeatures)._2.toArray

    var cumVar = 0.0
    for (i <- 0 to variances.size - 1) {
      cumVar += variances(i)
      println("%.5f, %.5f".format(variances(i), cumVar))
    }
  }

  def sumSqDist(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint]) {

    val numPc = 30
    var featureMatrix = new RowMatrix(labeledPointTrainRdd.map(_.features))
    val pcaModel = new PCA(numPc).fit(labeledPointTrainRdd.map(_.features))
    var pcMatrix = pcaModel.pc
    val projTrainPoints = featureMatrix.multiply(pcMatrix).rows

    projTrainPoints.cache()

    val kmWriter = new PrintWriter(new File("D:/Data/KDD99/results/kmeans_ssd.txt"))
    val bkWriter = new PrintWriter(new File("D:/Data/KDD99/results/bkmeans_ssd.txt"))
    val gmWriter = new PrintWriter(new File("D:/Data/KDD99/results/gmm_ssd.txt"))

    val MAX_CLUSTERS = 50

    for (k <- 1 to MAX_CLUSTERS) {
      var sumSqDist = 0.0

      val maxIterations = 10
      val kmModel = new KMeans().setK(k).setMaxIterations(maxIterations).run(projTrainPoints)

      projTrainPoints.collect().foreach { projTrainPoint =>

        val clusterIndex = kmModel.predict(projTrainPoint)
        val sqDist = Vectors.sqdist(projTrainPoint, kmModel.clusterCenters(clusterIndex))

        sumSqDist += sqDist

      }

      println(sumSqDist)
      kmWriter.println(sumSqDist)
      kmWriter.flush()
    }
    kmWriter.close()

    for (k <- 1 to MAX_CLUSTERS) {

      var sumSqDist = 0.0

      val bkModel = new BisectingKMeans().setK(k).run(projTrainPoints)

      projTrainPoints.collect().foreach { projTrainPoint =>

        val clusterIndex = bkModel.predict(projTrainPoint)
        val sqDist = Vectors.sqdist(projTrainPoint, bkModel.clusterCenters(clusterIndex))

        sumSqDist += sqDist

      }

      println(sumSqDist)
      bkWriter.println(sumSqDist)
      bkWriter.flush()
    }
    bkWriter.close()

    for (k <- 1 to MAX_CLUSTERS) {
      var sumSqDist = 0.0

      val gmModel = new GaussianMixture().setK(k).run(projTrainPoints)

      projTrainPoints.collect().foreach { projTrainPoint =>

        val clusterIndex = gmModel.predict(projTrainPoint)
        val sqDist = Vectors.sqdist(projTrainPoint, gmModel.gaussians(clusterIndex).mu)

        sumSqDist += sqDist

      }

      println(sumSqDist)
      gmWriter.println(sumSqDist)
      gmWriter.flush()
    }
    gmWriter.close()

  }

  def saveModels(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint]) {
    
    val numPc = 30
    val projTrainPoints = AdvUtil.getProjPointsAndPcMatrix(labeledPointTrainRdd.map(_.features), numPc)._1
    val pcMatrix = AdvUtil.getProjPointsAndPcMatrix(labeledPointTrainRdd.map(_.features), numPc)._2
    projTrainPoints.cache()

    sc.parallelize(Array(pcMatrix)).saveAsObjectFile(pcFile.format(numPc))

  }
  
  def saveKMeansModel(sc: SparkContext, projTrainPoints: RDD[Vector], numClusters: Int){

    val maxIterations = 10
    val kmModel = new KMeans().setK(numClusters).setMaxIterations(maxIterations).run(projTrainPoints)
    if (Path(kmModelDir.format(numClusters)).exists) Path(kmModelDir.format(numClusters)).deleteRecursively()
    kmModel.save(sc, kmModelDir.format(numClusters))

    var kmSqDistList: Map[Int, ListBuffer[Double]] = Map()

    for (i <- 0 to numClusters) {
      kmSqDistList += (i -> ListBuffer())
    }

    projTrainPoints.collect().foreach { projTrainPoint =>

      val clusterIndex = kmModel.predict(projTrainPoint)
      val sqDist = Vectors.sqdist(projTrainPoint, kmModel.clusterCenters(clusterIndex))

      kmSqDistList(clusterIndex) += sqDist
    }

    val kmWriter = new PrintWriter(new File(kmModelClusterFile.format(numClusters)))

    for (i <- 0 to numClusters - 1) {
      val mean = kmSqDistList(i).sum / kmSqDistList(i).size
      val normSumSqDiff = kmSqDistList(i).map(x => Math.pow(x - mean, 2)).sum / (kmSqDistList(i).size - 1)
      val stdDev = Math.sqrt(normSumSqDiff)
      val min = kmSqDistList(i).min
      val max = kmSqDistList(i).max

      kmWriter.println("%d,%d,%f,%f,%f,%f".format(i, kmSqDistList(i).size, min, max, mean, stdDev))
    }

    kmWriter.close()
  }
  
  def saveBKMeansModel(sc: SparkContext, projTrainPoints: RDD[Vector], numClusters: Int){

    val bkmModel = new BisectingKMeans().setK(numClusters).run(projTrainPoints)
    if (Path(bkmModelDir.format(numClusters)).exists) Path(bkmModelDir.format(numClusters)).deleteRecursively()
    bkmModel.save(sc, bkmModelDir.format(numClusters))

    var bkmSqDistList: Map[Int, ListBuffer[Double]] = Map()

    for (i <- 0 to numClusters) {
      bkmSqDistList += (i -> ListBuffer())
    }

    projTrainPoints.collect().foreach { projTrainPoint =>

      val clusterIndex = bkmModel.predict(projTrainPoint)
      val sqDist = Vectors.sqdist(projTrainPoint, bkmModel.clusterCenters(clusterIndex))

      bkmSqDistList(clusterIndex) += sqDist
    }

    val bkmWriter = new PrintWriter(new File(bkmModelClusterFile.format(numClusters)))

    for (i <- 0 to numClusters - 1) {
      val mean = bkmSqDistList(i).sum / bkmSqDistList(i).size
      val normSumSqDiff = bkmSqDistList(i).map(x => Math.pow(x - mean, 2)).sum / (bkmSqDistList(i).size - 1)
      val stdDev = Math.sqrt(normSumSqDiff)
      val min = bkmSqDistList(i).min
      val max = bkmSqDistList(i).max

      bkmWriter.println("%d,%d,%f,%f,%f,%f".format(i, bkmSqDistList(i).size, min, max, mean, stdDev))
    }

    bkmWriter.close()
  }
  
  def saveGmModel(sc: SparkContext, projTrainPoints: RDD[Vector], numClusters: Int){

    val gmModel = new GaussianMixture().setK(numClusters).run(projTrainPoints)
    if (Path(gmModelDir.format(numClusters)).exists) Path(gmModelDir.format(numClusters)).deleteRecursively()
    gmModel.save(sc, gmModelDir.format(numClusters))

    var gmSqDistList: Map[Int, ListBuffer[Double]] = Map()

    for (i <- 0 to numClusters) {
      gmSqDistList += (i -> ListBuffer())
    }

    projTrainPoints.collect().foreach { projTrainPoint =>

      val clusterIndex = gmModel.predict(projTrainPoint)
      val sqDist = Vectors.sqdist(projTrainPoint, gmModel.gaussians(clusterIndex).mu)

      gmSqDistList(clusterIndex) += sqDist
    }

    val gmWriter = new PrintWriter(new File(gmModelClusterFile.format(numClusters)))

    for (i <- 0 to numClusters - 1) {
      val mean = gmSqDistList(i).sum / gmSqDistList(i).size
      val normSumSqDiff = gmSqDistList(i).map(x => Math.pow(x - mean, 2)).sum / (gmSqDistList(i).size - 1)
      val stdDev = Math.sqrt(normSumSqDiff)
      val min = gmSqDistList(i).min
      val max = gmSqDistList(i).max

      gmWriter.println("%d,%d,%f,%f,%f,%f".format(i, gmSqDistList(i).size, min, max, mean, stdDev))
    }

    gmWriter.close()
  }

  def kMeansAnomalyDetectionResult(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numClusters: Int, numPc: Int) {

    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))
    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    val clusterInfoArray: Array[ClusterInfo] = getClusterInfoArray(sc, kmModelClusterFile.format(numClusters)) 

    var TP = 0
    var FP = 0
    var TN = 0
    var FN = 0
    
    val kmTpPointsWriter = new PrintWriter(new File(kmTpPointsFile.format(numClusters)))
    val kmTpClustersWriter = new PrintWriter(new File(kmTpClustersFile.format(numClusters)))
    

    labeledPointTestRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = kmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))

      if (sqDist > AdvUtil.chebyshevThreshold(clusterInfoArray(clusterIndex).meanSqDist, clusterInfoArray(clusterIndex).stdDevSqDist)) {
        if (label == 1.0){
          TP += 1
          kmTpPointsWriter.println(testPoint.features.toArray.mkString(","))
          
          val indexDistancePair = for(i <- 0 to kmModel.clusterCenters.size - 1) yield
            (i, Vectors.sqdist(projTestPoint, kmModel.clusterCenters(i)))
          val nearestClusters = indexDistancePair.sortWith((x,y) => x._2 < y._2).map(_._1)
          
          kmTpClustersWriter.println(nearestClusters.mkString(","))
          
        }
        else FP += 1
      } else {
        if (label == 1.0) FN += 1
        else TN += 1
      }
      
    }
    
    kmTpPointsWriter.close()
    kmTpClustersWriter.close()
    

    println("TP : " + TP)
    println("FP : " + FP)
    println("TN : " + TN)
    println("FN : " + FN)
    
    val precision = TP / (TP + FP).toDouble
    println("precision : %.3f".format(precision))
    
    val recall = TP / (TP + FN).toDouble
    println("Recall : %.3f".format(recall))
    
    println("Accuracy : %.3f".format((TP + TN) / (TP + TN + FP + FN).toFloat))
    
    println("F-Measure : %.3f".format((2 * precision * recall)/(precision + recall)))

  }
  
  def bkMeansAnomalyDetectionResult(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numClusters: Int, numPc: Int) {

    val bkmModel = BisectingKMeansModel.load(sc, bkmModelDir.format(numClusters))
    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    val clusterInfoArray: Array[ClusterInfo] = getClusterInfoArray(sc, bkmModelClusterFile.format(numClusters))

    var TP = 0
    var FP = 0
    var TN = 0
    var FN = 0
    
    val bkmTpPointsWriter = new PrintWriter(new File(bkmTpPointsFile.format(numClusters)))
    val bkmTpClustersWriter = new PrintWriter(new File(bkmTpClustersFile.format(numClusters)))
    

    labeledPointTestRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = bkmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, bkmModel.clusterCenters(clusterIndex))

      if (sqDist > AdvUtil.chebyshevThreshold(clusterInfoArray(clusterIndex).meanSqDist, clusterInfoArray(clusterIndex).stdDevSqDist)) {
        if (label == 1.0){
          TP += 1
          bkmTpPointsWriter.println(testPoint.features.toArray.mkString(","))
          
          val indexDistancePair = for(i <- 0 to bkmModel.clusterCenters.size - 1) yield
            (i, Vectors.sqdist(projTestPoint, bkmModel.clusterCenters(i)))
          val nearestClusters = indexDistancePair.sortWith((x,y) => x._2 < y._2).map(_._1)
          
          bkmTpClustersWriter.println(nearestClusters.mkString(","))
          
        }
        else FP += 1
      } else {
        if (label == 1.0) FN += 1
        else TN += 1
      }
      
    }
    
    bkmTpPointsWriter.close()
    bkmTpClustersWriter.close()
    

    println("TP : " + TP)
    println("FP : " + FP)
    println("TN : " + TN)
    println("FN : " + FN)
    
    val precision = TP / (TP + FP).toDouble
    println("precision : %.3f".format(precision))
    
    val recall = TP / (TP + FN).toDouble
    println("Recall : %.3f".format(recall))
    
    println("Accuracy : %.3f".format((TP + TN) / (TP + TN + FP + FN).toFloat))
    
    println("F-Measure : %.3f".format((2 * precision * recall)/(precision + recall)))

  }
  
  def gmAnomalyDetectionResult(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numClusters: Int, numPc: Int) {

    val gmModel = GaussianMixtureModel.load(sc, gmModelDir.format(numClusters))
    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    val clusterInfoArray: Array[ClusterInfo] = getClusterInfoArray(sc, gmModelClusterFile.format(numClusters)) 

    var TP = 0
    var FP = 0
    var TN = 0
    var FN = 0
    
    val gmTpPointsWriter = new PrintWriter(new File(gmTpPointsFile.format(numClusters)))
    val gmTpClustersWriter = new PrintWriter(new File(gmTpClustersFile.format(numClusters)))
    

    labeledPointTestRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = gmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, gmModel.gaussians(clusterIndex).mu)

      if (sqDist > AdvUtil.chebyshevThreshold(clusterInfoArray(clusterIndex).meanSqDist, clusterInfoArray(clusterIndex).stdDevSqDist)) {
        if (label == 1.0){
          TP += 1
          gmTpPointsWriter.println(testPoint.features.toArray.mkString(","))
          
          val indexDistancePair = for(i <- 0 to gmModel.gaussians.size - 1) yield
            (i, Vectors.sqdist(projTestPoint, gmModel.gaussians(i).mu))
          val nearestClusters = indexDistancePair.sortWith((x,y) => x._2 < y._2).map(_._1)
          
          gmTpClustersWriter.println(nearestClusters.mkString(","))
          
        }
        else FP += 1
      } else {
        if (label == 1.0) FN += 1
        else TN += 1
      }
      
    }
    
    gmTpPointsWriter.close()
    gmTpClustersWriter.close()
    

    println("TP : " + TP)
    println("FP : " + FP)
    println("TN : " + TN)
    println("FN : " + FN)
    
    val precision = TP / (TP + FP).toDouble
    println("precision : %.3f".format(precision))
    
    val recall = TP / (TP + FN).toDouble
    println("Recall : %.3f".format(recall))
    
    println("Accuracy : %.3f".format((TP + TN) / (TP + TN + FP + FN).toFloat))
    
    println("F-Measure : %.3f".format((2 * precision * recall)/(precision + recall)))

  }
  
  def getClusterInfoArray(sc: SparkContext, clusterFile: String): Array[ClusterInfo] = {
    
    val lines = sc.textFile(clusterFile).collect
    val clusterInfoArray: Array[ClusterInfo] = Array.ofDim[ClusterInfo](lines.size)
    
    lines.foreach { line =>
      val tokens = line.split(",", -1)
      val index = tokens(0).toInt
      val size = tokens(1).toInt
      val minSqDist = tokens(2).toDouble
      val maxSqDist = tokens(3).toDouble
      val meanSqDist = tokens(4).toDouble
      val stdDevSqDist = tokens(5).toDouble
      
      clusterInfoArray(index) = new ClusterInfo(index, size, minSqDist, maxSqDist, meanSqDist, stdDevSqDist)
    }
    
    clusterInfoArray
  }
  
 
  
}