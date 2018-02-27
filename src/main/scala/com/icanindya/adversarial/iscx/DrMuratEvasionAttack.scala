package com.icanindya.adversarial.iscx
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

object DrMuratEvasionAttack {

  val kmModelDir = "D:/Data/KDD99/dr_murat/model/kmeans/%d"
  val kmModelClusterFile = kmModelDir + "/clsuters.txt"

  val kmEvasionSuccessFile = "D:/Data/KDD99/dr_murat/evasion_results/kmeans/evasion_results.txt"

  
  var randomTestSamplesFile = "D:/Data/KDD99/final/random_test_%d"
  randomTestSamplesFile = "D:/Data/KDD99/final/test"
  
  val testSampleSize = 30000

  var pcMatrix: DenseMatrix = null
  var pcArray: Array[Array[Double]] = null
  var isAttrChangeable: Array[Double] = null

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    pcMatrix = sc.objectFile[DenseMatrix](DrMuratAnomalyDetection.pcFile.format(30)).collect()(0)
    pcArray = AdvUtil.pcToArr(pcMatrix)
    
//    println(pcMatrix.numRows + " " + pcMatrix.numCols)
    
    quickEvasionAttack(sc)
  }
  
  def quickEvasionAttack(sc: SparkContext){
    
    val labeledPointTestRdd = sc.textFile(randomTestSamplesFile.format(testSampleSize)).map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }

    labeledPointTestRdd.cache  
    
    var distThresArray = Array(1 * 0.31416, 2 * 0.31416 , 3 * 0.31416, 4 * 0.31416, 5 * 0.31416, 6 * 0.31416, 7 * 0.31416, 8 * 0.31416, 9 * 0.31416, 10 * 0.31416)
    
    var sqDistThres = Math.pow(5 * 0.31416, 2)
    var attrChangePercentage = 10 
    var numClusters = 5
    
    for(numClusters <- 5 to 25 by 5){
      kmQuickEvasionAttack(sc, labeledPointTestRdd, numClusters, sqDistThres, attrChangePercentage)
    }
    
    numClusters = 15
    attrChangePercentage = 10
    
    distThresArray = Array(1.26, 1.26 + 2 * 0.126 , 1.26 + 3 * 0.126, 1.26 + 4 * 0.126, 1.26 + 5 * 0.126, 1.26 + 6 * 0.126, 1.26 + 7 * 0.126, 1.26 + 8 * 0.126, 1.26 + 9 * 0.126, 1.26 + 10 * 0.126)
    val sqDistThresArray = distThresArray.map(x => Math.pow(x, 2))
    for(sqDistThres <- sqDistThresArray){
      kmQuickEvasionAttack(sc, labeledPointTestRdd, numClusters, sqDistThres, attrChangePercentage)
    }
    
    numClusters = 15
    sqDistThres = Math.pow(1.26 + 5 * 0.126, 2)    
    
    for(attrChangePercentage <- 2 to 40 by 2){
      kmQuickEvasionAttack(sc, labeledPointTestRdd, numClusters, sqDistThres, attrChangePercentage)
    }

  }
  
  def kmQuickEvasionAttack(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint], numClusters: Int, sqDistThres: Double, attrChangePercentage: Double) {
    
    var tp = 0
    var fp = 0
    var tn = 0
    var fn = 0

    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))
    
    
    labeledPointTestRdd.collect.foreach { testPoint =>
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
    
    println("tp = %d, tn = %d, fp = %d, fn = %d".format( tp, tn, fp, fn))
    
    val tpPoints = labeledPointTestRdd.filter { testPoint =>
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
    
    println("k: %d, t: %.2f, e: %.2f | tp = %d, tn = %d, fp = %d, fn = %d, ms = %d".format(numClusters, Math.sqrt(sqDistThres), attrChangePercentage, tp, tn, fp, fn, successCount))
    
    
    val pw = new PrintWriter(new FileWriter(kmEvasionSuccessFile, true))
    pw.println("k: %d, t: %.2f, e: %.2f | tp = %d, tn = %d, fp = %d, fn = %d, ms = %d".format(numClusters, Math.sqrt(sqDistThres), attrChangePercentage, tp, tn, fp, fn, successCount))
    pw.close
    
  
  }

  def getKmClusterOptDistances(sc: SparkContext, tpPoints: Array[Vector], numClusters: Int, sqDistThres: Double, attrChangePercentage: Double): Array[Double] = {
    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))

    val clusters = kmModel.clusterCenters

    val optDistances = for (i <- 0 to clusters.size - 1) yield {
      val optLocation = getOptimumLocation(clusters(i), tpPoints, attrChangePercentage, sqDistThres)
      val optIndex = optLocation._1
      val optDistance = optLocation._2
      println("%d, %.2f, %.2f cluster %d : %d".format(numClusters, sqDistThres, attrChangePercentage, i, optIndex))
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
      Array.fill(tpPoints(0).size)(attrChangePercentage))

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
    
    val attrChangeThresholds = Array.fill(tpPoints(0).size)(attrChangePercentage)
    
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


  