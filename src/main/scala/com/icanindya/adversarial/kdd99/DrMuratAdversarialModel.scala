package com.icanindya.adversarial.kdd99

import com.icanindya.adversarial.Spark
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import com.icanindya.adversarial.AdvUtil
import com.icanindya.adversarial.ClusterInfo
import scala.io.Source
import com.icanindya.adversarial.CplexOptimizer
import com.icanindya.adversarial.CplexSolution
import java.io.PrintWriter
import java.io.File
import org.apache.spark.mllib.clustering.BisectingKMeansModel
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import java.io.FileWriter

object DrMuratAdversarialModel {

  val kmModelDir = DrMuratAnomalyModel.kmModelDir
  val bkmModelDir = DrMuratAnomalyModel.bkmModelDir

  val kmEvasionResultsFile = "D:/Data/KDD99/dr_murat/evasion_results/kmeans_evasion_results.txt"
  val bkmEvasionResultsFile = "D:/Data/KDD99/dr_murat/evasion_results/bkm_evasion_results.txt"
  
  var pcMatrix: DenseMatrix = null
  var pcArray: Array[Array[Double]] = null
  var isAttrChangeable: Array[Double] = null
  
  val fpWeight = 0.1

  def main(args: Array[String]) {
    
    val sc = Spark.getContext()
    pcMatrix = sc.objectFile[DenseMatrix](DrMuratAnomalyModel.pcFile.format(30)).collect()(0)
    pcArray = AdvUtil.pcToArr(pcMatrix)
    quickEvasionAttack(sc)
  }
  
  def quickEvasionAttack(sc: SparkContext){
    
    val labeledPointValidRdd = sc.textFile(FinalDataset.finalValidationFile).map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }

    labeledPointValidRdd.cache  
    
    var distThresArray = for(i <- 5 to 20) yield i * 0.1

    for(attrChangePercentage <- 5 to 25 by 5){
      for(numClusters <- 5 to 25 by 5){
        for(distThres <- distThresArray){
          kmQuickEvasionAttack(sc, labeledPointValidRdd, numClusters, Math.pow(distThres, 2), attrChangePercentage)
        }
      }
    }

  }
  
  def kmQuickEvasionAttack(sc: SparkContext, labeledPointValidRdd: RDD[LabeledPoint], numClusters: Int, sqDistThres: Double, attrChangePercentage: Double) {
    
    var (tp, tn, fp, fn) = (0, 0, 0, 0)

    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))
    
    labeledPointValidRdd.collect.foreach { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = kmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))
    
      if (sqDist > sqDistThres) {
        if (label == 1.0) tp += 1
        else fp += 1
      } else {
        if (label == 1.0) fn += 1
        else tn += 1
      }
    }
    
    
    val tpPoints = labeledPointValidRdd.filter { testPoint =>
      val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
      val label = testPoint.label

      val clusterIndex = kmModel.predict(projTestPoint)
      val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))
    
      label == 1.0 && sqDist > sqDistThres
    }
    .map(_.features).collect
    
    val sortedClusters = for(tpPoint <- tpPoints) yield{
      val projTpPoint = AdvUtil.getProjFromOrig(tpPoint, pcMatrix)
      val indexDistancePairs = for(i <- 0 to kmModel.clusterCenters.size - 1) yield
              (i, Vectors.sqdist(projTpPoint, kmModel.clusterCenters(i)))
      val sortedPairs = indexDistancePairs.sortWith((x,y) => x._2 < y._2).map(_._1)  
      sortedPairs.toArray
    }

    val clusterOptDistance = getKmClusterOptDistances(sc, tpPoints, numClusters, sqDistThres, attrChangePercentage)
      
    val successCount = launchEvasionAttack(tpPoints, sortedClusters, kmModel.clusterCenters, sqDistThres, clusterOptDistance, attrChangePercentage)
    
    val expressionValue = fpWeight * fp + (1 - fpWeight) * (fn + successCount)
    
    println("e = %3.2f, k = %2d, t = %.2f : tp = %4d, tn = %4d, fp = %4d, fn = %4d, ms = %4d, value = %7.2f".format(attrChangePercentage, numClusters, Math.sqrt(sqDistThres), tp, tn, fp, fn, successCount, expressionValue))
    
    
    val pw = new PrintWriter(new FileWriter(kmEvasionResultsFile, true))
    pw.println("e = %3.2f, k = %2d, t = %.2f : tp = %4d, tn = %4d, fp = %4d, fn = %4d, ms = %4d, value = %7.2f".format(attrChangePercentage, numClusters, Math.sqrt(sqDistThres), tp, tn, fp, fn, successCount, expressionValue))
    pw.close
    
  
  }

  def getKmClusterOptDistances(sc: SparkContext, tpPoints: Array[Vector], numClusters: Int, sqDistThres: Double, attrChangePercentage: Double): Array[Double] = {
    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))

    val clusters = kmModel.clusterCenters

    val optDistances = for (i <- 0 to clusters.size - 1) yield {
      val optLocation = getOptimumLocation(clusters(i), tpPoints, attrChangePercentage, sqDistThres)
      val optIndex = optLocation._1
      val optDistance = optLocation._2
      println("%d, %.2f, %.2f cluster %d : %d".format(numClusters, Math.sqrt(sqDistThres), attrChangePercentage, i, optIndex))
      optDistance
    }
    
    optDistances.toArray
  }

  def getOptimumLocation(cluster: Vector, tpPoints: Array[Vector], attrChangePercentage: Double, sqDistThres: Double): (Int, Double) = {
    val sortedTpPoints = tpPoints.map { origPoint =>
      val projPoint = AdvUtil.getProjFromOrig(origPoint, pcMatrix)
      val sqDist = Vectors.sqdist(projPoint, cluster)
      (origPoint, projPoint, sqDist)
    }
    .sortWith((x, y) => x._3 < y._3).map(x => (x._1, x._2))

    val optDistanceIndex = binarySearchForSolution(cluster, sqDistThres,
      sortedTpPoints, pcMatrix,
      AttrChangability.getAttrChangeThresholds(attrChangePercentage))

    val optDistance = Vectors.sqdist(sortedTpPoints(optDistanceIndex)._2, cluster)

    (optDistanceIndex, optDistance)
  }

  def binarySearchForSolution(cluster: Vector, sqDistThres: Double,
                              sortedTpPoints: Array[(Vector, Vector)], pcMatrix: DenseMatrix,
                              attrChangeThres: Array[Double]): Int = {
    
//    println("attr change: " + attrChangeThres.mkString(","))

    var left = 0
    var right = sortedTpPoints.size - 1

    while (left < right - 1) {

      //      println("left: %d, right: %d".format(left, right))

      var middle = Math.floor((left + right) / 2).toInt

      val targetTpPoint = sortedTpPoints(middle)

      val cplexSolution = CplexOptimizer.softOptimize(targetTpPoint._1.toArray, targetTpPoint._2.toArray,
        cluster.toArray, sqDistThres,
        AdvUtil.pcToArr(pcMatrix), attrChangeThres)

      if (cplexSolution.found) {
        left = middle
      } else {
//        println("solution not found at " + middle)
//        println("target tp point proj: " + targetTpPoint._2.toArray.mkString(","))
//        println("target cluster center: " + cluster.toArray.mkString(","))
//        println("distance: " + Math.sqrt(Vectors.sqdist(targetTpPoint._2, cluster)))
        right = middle - 1
      }
    }
    left
  }

  def launchEvasionAttack(tpPoints: Array[Vector], sortedClusters: Array[Array[Int]],
        centers: Array[Vector], sqDistThres: Double,
        clusterOptDistance: Array[Double], attrChangePercentage: Double) : Int = {
    
    val attrChangeThresholds = AttrChangability.getAttrChangeThresholds(attrChangePercentage)
    
    var successCount = 0
    
    for (i <- 0 to tpPoints.size - 1) {

      var cplexSolution = new CplexSolution()

      for {
        j <- 0 to sortedClusters(i).size - 1
        if !cplexSolution.found
      } {

        val origX = tpPoints(i).toArray
        val projX = AdvUtil.getProjFromOrig(tpPoints(i), pcMatrix).toArray
        val clusterIndex = sortedClusters(i)(j)
        val projCenter = centers(clusterIndex).toArray

        val distanceFromCluster = Vectors.sqdist(Vectors.dense(projX), Vectors.dense(projCenter))

        if (distanceFromCluster <= clusterOptDistance(clusterIndex)) {

          
          cplexSolution = CplexOptimizer.softOptimize(origX, projX, projCenter, sqDistThres, pcArray, attrChangeThresholds)

          if (cplexSolution.found){
            successCount += 1
            
//            println("Evasion at %.2f: try = %d, success = %d".format(attrChangePercentage, i, successCount))
            
          }
        }
      }
    }
    
    successCount
    
  }

}


  