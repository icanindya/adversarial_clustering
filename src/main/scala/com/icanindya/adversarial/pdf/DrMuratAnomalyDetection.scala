package com.icanindya.adversarial.pdf

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

object DrMuratAnomalyDetection {

  val kmModelDir = "D:/Data/PDF/dr_murat/model/kmeans/%d"
  val kmModelClusterStatFile = kmModelDir + "/clsuters.txt"
  val kmAnomalyResultFilePath = "D:/Data/PDF/dr_murat/anomaly_results/kmeans/k_%d_t_%.2f.txt"
  val pcFile = "D:/Data/PDF/dr_murat/model/pc/%d"
  
  val testSampleSize = 30000

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
    

//    for(k <- 5 to 25 by 5){
//      saveKMeansModel(sc, labeledPointTrainRdd, 60, k)
//    }
    
    for(k <- 5 to 25 by 5){
      for(c <- 1 to 10){
        val sqDistThres = Math.pow(c * 0.313, 2)
        saveKMeansAnomalyDetectionResult(sc, labeledPointTestRdd, 60, k, sqDistThres)
      }
    }
    
  }
  
   def pcVariaance(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint]) {
    val numFeatures = labeledPointTrainRdd.take(1)(0).features.size
    var mat = new RowMatrix(labeledPointTrainRdd.map(_.features))

    val variances = mat.computePrincipalComponentsAndExplainedVariance(numFeatures)._2.toArray

    var cumVar = 0.0
    for (i <- 0 to variances.size - 1) {
      cumVar += variances(i)
      println("%d, %.5f, %.5f".format(i + 1, variances(i), cumVar))
    }
  }

  def saveKMeansModel(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint], numPc: Int, numClusters: Int) {

    val pcMatrix = AdvUtil.getPcMatrix(labeledPointTrainRdd.map(_.features), numPc)
    val projTrainPoints = AdvUtil.getProjPoints(labeledPointTrainRdd.map(_.features), pcMatrix)
    projTrainPoints.cache()

    sc.parallelize(Array(pcMatrix)).saveAsObjectFile(pcFile.format(numPc))
    
    val kmModel = new KMeans().setK(numClusters).run(projTrainPoints)
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

    val kmWriter = new PrintWriter(new File(kmModelClusterStatFile.format(numClusters)))

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

  def saveKMeansAnomalyDetectionResult(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numPc: Int, numClusters: Int, sqDistThres: Double) {
    

    val kmAnomalyResultFile = new File(kmAnomalyResultFilePath.format(numClusters, Math.sqrt(sqDistThres)))
    
    kmAnomalyResultFile.getParentFile.mkdirs()
    
    val anomalyResultWriter = new PrintWriter(kmAnomalyResultFile)
    
    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))
    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    var TP = 0
    var FP = 0
    var TN = 0
    var FN = 0

    labeledPointTestRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = kmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))

      if (sqDist > sqDistThres) {
        if (label == 1.0) {
          TP += 1
        } else FP += 1
      } else {
        if (label == 1.0) FN += 1
        else TN += 1
      }

    }

    println("TP : " + TP)
    anomalyResultWriter.println("TP : " + TP)
    println("FP : " + FP)
    anomalyResultWriter.println("FP : " + FP)
    println("TN : " + TN)
    anomalyResultWriter.println("TN : " + TN)
    println("FN : " + FN)
    anomalyResultWriter.println("FN : " + FN)

    val precision = TP / (TP + FP).toDouble
    println("precision : %.3f".format(precision))
    anomalyResultWriter.println("precision : %.3f".format(precision))
    
    val recall = TP / (TP + FN).toDouble
    println("Recall : %.3f".format(recall))
    anomalyResultWriter.println("Recall : %.3f".format(recall))
    
    println("Accuracy : %.3f".format((TP + TN) / (TP + TN + FP + FN).toFloat))
    anomalyResultWriter.println("Accuracy : %.3f".format((TP + TN) / (TP + TN + FP + FN).toFloat))
    
    println("F-Measure : %.3f".format((2 * precision * recall) / (precision + recall)))
    anomalyResultWriter.println("F-Measure : %.3f".format((2 * precision * recall) / (precision + recall)))
    
    anomalyResultWriter.close()
  }

}