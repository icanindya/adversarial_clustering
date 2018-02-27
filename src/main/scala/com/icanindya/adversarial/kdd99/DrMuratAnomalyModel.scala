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
import org.apache.spark.mllib.clustering.BisectingKMeans

object DrMuratAnomalyModel {

  val kmModelDir = "D:/Data/KDD99/dr_murat/model/kmeans/%d"
  val kmModelClusterStatFile = kmModelDir + "/clsuters.txt"
  val kmAnomalyResultFilePath = "D:/Data/KDD99/dr_murat/anomaly_results/kmeans/k_%d_t_%.2f.txt"
  
  val bkmModelDir = "D:/Data/KDD99/dr_murat/model/bkm/%d"
  val bkmModelClusterStatFile = bkmModelDir + "/clsuters.txt"
  val bkmAnomalyResultFilePath = "D:/Data/KDD99/dr_murat/anomaly_results/bkm/k_%d_t_%.2f.txt"
  
  
  val pcFile = "D:/Data/KDD99/dr_murat/model/pc/%d"
  val NUM_PC = 30
  
  
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

    saveBKMeansModels(sc, labeledPointTrainRdd, NUM_PC)
    
    
  }
  
  def saveKMeansModels(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint], numPc: Int){
     for(k <- 5 to 25 by 5){
      saveKMeansModel(sc, labeledPointTrainRdd, NUM_PC, k)
    }
  }
  
  def saveBKMeansModels(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint], numPc: Int){
     for(k <- 5 to 25 by 5){
      saveBKMeansModel(sc, labeledPointTrainRdd, NUM_PC, k)
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

    var kmDistList: Map[Int, ListBuffer[Double]] = Map()

    for (i <- 0 to numClusters) {
      kmDistList += (i -> ListBuffer())
    }

    projTrainPoints.collect().foreach { projTrainPoint =>

      val clusterIndex = kmModel.predict(projTrainPoint)
      val dist = Math.sqrt(Vectors.sqdist(projTrainPoint, kmModel.clusterCenters(clusterIndex)))

      kmDistList(clusterIndex) += dist
    }

    val kmWriter = new PrintWriter(new File(kmModelClusterStatFile.format(numClusters)))

    for (i <- 0 to numClusters - 1) {
      
      val mean = kmDistList(i).sum / kmDistList(i).size
      val normSumDiff = kmDistList(i).map(x => Math.pow(x - mean, 2)).sum / (kmDistList(i).size - 1)
      val stdDev = Math.sqrt(normSumDiff)
      val min = kmDistList(i).min
      val max = kmDistList(i).max
      

      kmWriter.println("%d,%d,%f,%f,%f,%f".format(i, kmDistList(i).size, min, max, mean, stdDev))
    }

    kmWriter.close()
  }
  
  def saveBKMeansModel(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint], numPc: Int, numClusters: Int) {

    val pcMatrix = AdvUtil.getPcMatrix(labeledPointTrainRdd.map(_.features), numPc)
    val projTrainPoints = AdvUtil.getProjPoints(labeledPointTrainRdd.map(_.features), pcMatrix)
    projTrainPoints.cache()

    sc.parallelize(Array(pcMatrix)).saveAsObjectFile(pcFile.format(numPc))
    
    val bkmModel = new BisectingKMeans().setK(numClusters).run(projTrainPoints)
    if (Path(bkmModelDir.format(numClusters)).exists) Path(bkmModelDir.format(numClusters)).deleteRecursively()
    bkmModel.save(sc, bkmModelDir.format(numClusters))

    var bkmDistList: Map[Int, ListBuffer[Double]] = Map()

    for (i <- 0 to numClusters) {
      bkmDistList += (i -> ListBuffer())
    }

    projTrainPoints.collect().foreach { projTrainPoint =>

      val clusterIndex = bkmModel.predict(projTrainPoint)
      val dist = Math.sqrt(Vectors.sqdist(projTrainPoint, bkmModel.clusterCenters(clusterIndex)))

      bkmDistList(clusterIndex) += dist
    }

    val bkmWriter = new PrintWriter(new File(bkmModelClusterStatFile.format(numClusters)))

    for (i <- 0 to numClusters - 1) {
      
      val mean = bkmDistList(i).sum / bkmDistList(i).size
      val normSumDiff = bkmDistList(i).map(x => Math.pow(x - mean, 2)).sum / (bkmDistList(i).size - 1)
      val stdDev = Math.sqrt(normSumDiff)
      val min = bkmDistList(i).min
      val max = bkmDistList(i).max
      

      bkmWriter.println("%d,%d,%f,%f,%f,%f".format(i, bkmDistList(i).size, min, max, mean, stdDev))
    }

    bkmWriter.close()
  }

  def saveKMeansAnomalyDetectionResult(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numPc: Int, numClusters: Int, sqDistThres: Double) {
    

    val kmAnomalyResultFile = new File(kmAnomalyResultFilePath.format(numClusters, Math.sqrt(sqDistThres)))
    
    kmAnomalyResultFile.getParentFile.mkdirs()
    
    val anomalyResultWriter = new PrintWriter(kmAnomalyResultFile)
    
    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))
    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    var tp = 0
    var fp = 0
    var tn = 0
    var fn = 0

    labeledPointTestRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = kmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))

      if (sqDist > sqDistThres) {
        if (label == 1.0) {
          tp += 1
        } else fp += 1
      } else {
        if (label == 1.0) fn += 1
        else tn += 1
      }

    }

    println("TP : " + tp)
    anomalyResultWriter.println("TP : " + tp)
    println("FP : " + fp)
    anomalyResultWriter.println("FP : " + fp)
    println("TN : " + tn)
    anomalyResultWriter.println("TN : " + tn)
    println("FN : " + fn)
    anomalyResultWriter.println("FN : " + fn)

    val precision = tp / (tp + fp).toDouble
    println("precision : %.3f".format(precision))
    anomalyResultWriter.println("precision : %.3f".format(precision))
    
    val recall = tp / (tp + fn).toDouble
    println("Recall : %.3f".format(recall))
    anomalyResultWriter.println("Recall : %.3f".format(recall))
    
    println("Accuracy : %.3f".format((tp + tn) / (tp + tn + fp + fn).toFloat))
    anomalyResultWriter.println("Accuracy : %.3f".format((tp + tn) / (tp + tn + fp + fn).toFloat))
    
    println("F-Measure : %.3f".format((2 * precision * recall) / (precision + recall)))
    anomalyResultWriter.println("F-Measure : %.3f".format((2 * precision * recall) / (precision + recall)))
    
    anomalyResultWriter.close()
  }

}