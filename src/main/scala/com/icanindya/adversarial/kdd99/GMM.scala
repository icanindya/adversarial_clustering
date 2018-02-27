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
import org.apache.spark.mllib.clustering.GaussianMixture

object GMM {

  val gmModelDir = "D:/Data/KDD99/dr_murat/model/gmm/%d"
  val gmModelClusterStatFile = gmModelDir + "/clsuters.txt"
  val gmAnomalyResultFilePath = "D:/Data/KDD99/dr_murat/anomaly_results/gmm/k_%d_p_%.2f.txt"
  val pcFile = "D:/Data/KDD99/dr_murat/model/pc/%d"

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
    //      runGmModel(sc, labeledPointTrainRdd, 30, k)
    //    }

    val requiredProb = 0.05

    for (k <- 5 to 25 by 5) {
      saveGmAnomalyDetectionResult(sc, labeledPointTestRdd, 30, k, requiredProb)
    }

  }

  def runGmModel(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint], numPc: Int, numClusters: Int) {

    val pcMatrix = AdvUtil.getPcMatrix(labeledPointTrainRdd.map(_.features), numPc)
    val projTrainPoints = AdvUtil.getProjPoints(labeledPointTrainRdd.map(_.features), pcMatrix)
    projTrainPoints.cache()

    sc.parallelize(Array(pcMatrix)).saveAsObjectFile(pcFile.format(numPc))

    val gmModel = new GaussianMixture().setK(numClusters).run(projTrainPoints)
    if (Path(gmModelDir.format(numClusters)).exists) Path(gmModelDir.format(numClusters)).deleteRecursively()
    gmModel.save(sc, gmModelDir.format(numClusters))

    //    projTrainPoints.collect().foreach { projTrainPoint =>
    //
    //      val clusterIndex = gmModel.predict(projTrainPoint)
    //      
    //      println(clusterIndex + " vs " + gmModel.predictSoft(projTrainPoint).mkString(","))
    //
    //    }
  }

  def saveGmAnomalyDetectionResult(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numPc: Int, numClusters: Int, requiredProb: Double) {

    val gmAnomalyResultFile = new File(gmAnomalyResultFilePath.format(numClusters, Math.sqrt(requiredProb)))

    gmAnomalyResultFile.getParentFile.mkdirs()

    val anomalyResultWriter = new PrintWriter(gmAnomalyResultFile)

    val gmModel = GaussianMixtureModel.load(sc, gmModelDir.format(numClusters))
    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    var tp = 0
    var fp = 0
    var tn = 0
    var fn = 0

    labeledPointTestRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = gmModel.predict(projTestPoint)

      val highestProb = gmModel.predictSoft(projTestPoint)(clusterIndex)

      if (highestProb * 0.9 <= 1.0 / numClusters) {
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

  }

}