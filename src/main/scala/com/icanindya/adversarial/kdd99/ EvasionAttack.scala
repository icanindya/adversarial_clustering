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

object EvasionAttack {

  val kmModelDir = "D:/Data/KDD99/model/kmeans/%d"
  val kmModelClusterFile = kmModelDir + "/clsuters.txt"
  val bkmModelDir = "D:/Data/KDD99/model/bkmeans/%d"
  val bkmModelClusterFile = bkmModelDir + "/clsuters.txt"
  val gmModelDir = "D:/Data/KDD99/model/gm/%d"
  val gmModelClusterFile = gmModelDir + "/clsuters.txt"

  val pcFile = "D:/Data/KDD99/model/pc/%d"

  val kmTpPointsFile = kmModelDir + "/tp_points.txt"
  val kmTpClustersFile = kmModelDir + "/tp_clusters.txt"
  val bkmTpPointsFile = bkmModelDir + "/tp_points.txt"
  val bkmTpClustersFile = bkmModelDir + "/tp_clusters.txt"
  val gmTpPointsFile = gmModelDir + "/tp_points.txt"
  val gmTpClustersFile = gmModelDir + "/tp_clusters.txt"

  val attrChangeRangeFile = "D:/Data/KDD99/attr_change_ranges.txt"

  val kmClusterOptDistanceFile = kmModelDir + "/clsuter_opt_dist_%.2f.txt"
  val bkmClusterOptDistanceFile = bkmModelDir + "/clsuter_opt_dist_%.2f.txt"
  val gmClusterOptDistanceFile = gmModelDir + "/clsuter_opt_dist_%.2f.txt"
  
  val kmEvasionSuccessFile = kmModelDir + "/evasion_success.txt"
  val bkmEvasionSuccessFile = bkmModelDir + "/evasion_success.txt"
  val gmEvasionSuccessFile = gmModelDir + "/evasion_success.txt"

  val NUM_CONSIDERED_CLUSTERS = 5
  val NUM_PC = 30

  var pcMatrix: DenseMatrix = null
  var pcArray: Array[Array[Double]] = null
  var isAttrChangeable: Array[Double] = null

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(NUM_PC)).collect()(0)
    pcArray = AdvUtil.pcToArr(pcMatrix)
    
   quickEvasionAttack(sc)
//    saveClusterOpt(sc)
  }
  
  def quickEvasionAttack(sc: SparkContext){
    
    val kmNumClusters = 10
    
    val kmPercentageSuccess = for(attrChangePercentage <- 12 to 20 by 2) yield 
    kmQuickEvasionAttack(sc, kmNumClusters, attrChangePercentage)
    
    val kmPw = new PrintWriter(new File(kmEvasionSuccessFile.format(kmNumClusters)))
    kmPw.println(kmPercentageSuccess.map(x => "%.2f,%d".format(x._1, x._2)).mkString("\r\n"))
    kmPw.close()
    
    val bkmNumClusters = 10
    
    val bkmPercentageSuccess = for(attrChangePercentage <- 12 to 20 by 2) yield 
    bkmQuickEvasionAttack(sc, bkmNumClusters, attrChangePercentage)
    
    val bkmPw = new PrintWriter(new File(bkmEvasionSuccessFile.format(bkmNumClusters)))
    bkmPw.println(bkmPercentageSuccess.map(x => "%.2f,%d".format(x._1, x._2)).mkString("\r\n"))
    bkmPw.close()
    
//    val gmNumClusters = 24
//    
//    val percentageSuccess = for(attrChangePercentage <- 2 to 10 by 2) yield 
//    gmQuickEvasionAttack(sc, gmNumClusters, attrChangePercentage)
//    
//    val gmPw = new PrintWriter(new File(gmEvasionSuccessFile.format(gmNumClusters)))
//    gmPw.println(percentageSuccess.map(x => "%.2f,%d".format(x._1, x._2)).mkString("\r\n"))
//    gmPw.close()
//    
  }
  
  def saveClusterOpt(sc: SparkContext){
    for (attrChangePercentage <- 12 to 20 by 2) {
      saveKmClusterOpt(sc, 10, NUM_PC, attrChangePercentage.toDouble)
      saveBkmClusterOpt(sc, 10, NUM_PC, attrChangePercentage.toDouble)
//      saveGmClusterOpt(sc, 24, NUM_PC, attrChangePercentage.toDouble)
    }
  }

  def saveKmClusterOpt(sc: SparkContext, numClusters: Int, numPc: Int, attrChangePercentage: Double) {
    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))
    val tpPoints = sc.textFile(kmTpPointsFile.format(numClusters))
      .map { x =>
        Vectors.dense(x.split(",", -1).map(_.toDouble))
      }
      .collect()

    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    val clusters = kmModel.clusterCenters
    val clusterInfoArray = AnomalyDetection.getClusterInfoArray(sc, kmModelClusterFile.format(numClusters))

    val optDistances = for (i <- 0 to clusters.size - 1) yield {
      val optLocation = getOptimumLocation(clusters(i), clusterInfoArray(i), tpPoints, attrChangePercentage)
      val optIndex = optLocation._1
      val optDistance = optLocation._2
      println("kMeans cluster %d at %.2f: %d".format(i, attrChangePercentage, optIndex))
      optDistance
    }

    val file = new File(kmClusterOptDistanceFile.format(numClusters, attrChangePercentage))
    file.getParentFile.mkdirs()
    val pw = new PrintWriter(file)
    pw.println(optDistances.mkString(","))
    pw.close()
  }

  def saveBkmClusterOpt(sc: SparkContext, numClusters: Int, numPc: Int, attrChangePercentage: Double) {
    val bkmModel = BisectingKMeansModel.load(sc, bkmModelDir.format(numClusters))
    val tpPoints = sc.textFile(bkmTpPointsFile.format(numClusters))
      .map { x =>
        Vectors.dense(x.split(",", -1).map(_.toDouble))
      }
      .collect()

    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    val clusters = bkmModel.clusterCenters
    val clusterInfoArray = AnomalyDetection.getClusterInfoArray(sc, bkmModelClusterFile.format(numClusters))

    val optDistances = for (i <- 0 to clusters.size - 1) yield {
      val optLocation = getOptimumLocation(clusters(i), clusterInfoArray(i), tpPoints, attrChangePercentage)
      val optIndex = optLocation._1
      val optDistance = optLocation._2
      println("BKMeans cluster %d at %.2f: %d".format(i, attrChangePercentage, optIndex))
      optDistance
    }

    val file = new File(bkmClusterOptDistanceFile.format(numClusters, attrChangePercentage))
    file.getParentFile.mkdirs()
    val pw = new PrintWriter(file)
    pw.println(optDistances.mkString(","))
    pw.close()
  }

  def saveGmClusterOpt(sc: SparkContext, numClusters: Int, numPc: Int, attrChangePercentage: Double) {
    val gmModel = GaussianMixtureModel.load(sc, gmModelDir.format(numClusters))
    val tpPoints = sc.textFile(gmTpPointsFile.format(numClusters))
      .map { x =>
        Vectors.dense(x.split(",", -1).map(_.toDouble))
      }
      .collect()

    val pcMatrix = sc.objectFile[DenseMatrix](pcFile.format(numPc)).collect()(0)

    val clusters = gmModel.gaussians.map(_.mu)

    gmModel.gaussians(0).sigma

    val clusterInfoArray = AnomalyDetection.getClusterInfoArray(sc, gmModelClusterFile.format(numClusters))

    val optDistances = for (i <- 0 to clusters.size - 1) yield {
      val optLocation = getOptimumLocation(clusters(i), clusterInfoArray(i), tpPoints, attrChangePercentage)
      val optIndex = optLocation._1
      val optDistance = optLocation._2
      println("GM cluster %d at %.2f: %d".format(i, attrChangePercentage, optIndex))
      optDistance
    }

    val file = new File(gmClusterOptDistanceFile.format(numClusters, attrChangePercentage))
    file.getParentFile.mkdirs()
    val pw = new PrintWriter(file)
    pw.println(optDistances.mkString(","))
    pw.close()
  }

  def getOptimumLocation(cluster: Vector, clusterInfo: ClusterInfo, tpPoints: Array[Vector], attrChangePercentage: Double): (Int, Double) = {
    val sortedTpPoints = tpPoints.map { origPoint =>
      val projPoint = AdvUtil.getProjFromOrig(origPoint, pcMatrix)
      val sqDist = Vectors.sqdist(projPoint, cluster)
      (origPoint, projPoint, sqDist)
    }
      .sortWith((x, y) => x._3 < y._3).map(x => (x._1, x._2))

    val optDistanceIndex = binarySearchForSolution(cluster, AdvUtil.chebyshevThreshold(clusterInfo.meanSqDist, clusterInfo.stdDevSqDist),
      sortedTpPoints, pcMatrix,
      AttrChangability.getAttrChangeThresholds(attrChangePercentage))

    val optDistance = Vectors.sqdist(sortedTpPoints(optDistanceIndex)._2, cluster)

    (optDistanceIndex, optDistance)
  }

  def binarySearchForSolution(cluster: Vector, sqDistThres: Double,
                              sortedTpPoints: Array[(Vector, Vector)], pcMatrix: DenseMatrix,
                              attrChangeThres: Array[Double]): Int = {

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
      } else right = middle - 1
    }
    left
  }

  def kmEvationAttack(sc: SparkContext, attrChangePercentage: Double) {
    
     val attrChangeThresholds = AttrChangability.getAttrChangeThresholds(attrChangePercentage)
    

    val numClusters = 10

    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))

    val clusterInfoArray: Array[ClusterInfo] = AnomalyDetection.getClusterInfoArray(sc, kmModelClusterFile.format(numClusters))

    val tpPoints = Source.fromFile(kmTpPointsFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toDouble)).toArray

    val tpClusters = Source.fromFile(kmTpClustersFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toInt).take(NUM_CONSIDERED_CLUSTERS)).toArray

    for (i <- 0 to tpPoints.size - 1) {

      var bestCplexSolution = new CplexSolution()

      for (j <- 0 to tpClusters(i).size - 1) {

        val origX = tpPoints(i)
        val projX = AdvUtil.getProjFromOrig(Vectors.dense(tpPoints(i)), pcMatrix).toArray
        val clusterIndex = tpClusters(i)(j)
        val projCenter = kmModel.clusterCenters(clusterIndex).toArray
        val sqDistThres = AdvUtil.chebyshevThreshold(clusterInfoArray(clusterIndex).meanSqDist, clusterInfoArray(clusterIndex).stdDevSqDist)

        val cplexSolution = CplexOptimizer.hardOptimize(origX, projX, projCenter, sqDistThres, pcArray, attrChangeThresholds)

        if (cplexSolution.found && cplexSolution.objValue < bestCplexSolution.objValue) {

          bestCplexSolution = cplexSolution

          println("solution found")

        }
      }
      println("trial %d:, %s".format(i, if (bestCplexSolution.found) "success" else "failure"))
    }
  }
  
  def kmQuickEvasionAttack(sc: SparkContext, numClusters: Int, attrChangePercentage: Double): (Double, Int) = {

    val kmModel = KMeansModel.load(sc, kmModelDir.format(numClusters))

    val clusterInfoArray: Array[ClusterInfo] = AnomalyDetection.getClusterInfoArray(sc, kmModelClusterFile.format(numClusters))

    val tpPoints = Source.fromFile(kmTpPointsFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toDouble)).toArray

    val sortedClusters = Source.fromFile(kmTpClustersFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toInt)).toArray

    val clusterOptDistance = Source.fromFile(kmClusterOptDistanceFile.format(numClusters, attrChangePercentage)).getLines().next()
      .split(",", -1).map(_.toDouble)
      
    val successCount = launchEvasionAttack(tpPoints, sortedClusters, kmModel.clusterCenters, clusterInfoArray, clusterOptDistance, attrChangePercentage)
    
    println("KMeans evasion success at %.2f: %d".format(attrChangePercentage, successCount))
    
    (attrChangePercentage, successCount)
  }
  
  def bkmQuickEvasionAttack(sc: SparkContext, numClusters: Int, attrChangePercentage: Double): (Double, Int) = {

    val bkmModel = BisectingKMeansModel.load(sc, bkmModelDir.format(numClusters))

    val clusterInfoArray: Array[ClusterInfo] = AnomalyDetection.getClusterInfoArray(sc, bkmModelClusterFile.format(numClusters))

    val tpPoints = Source.fromFile(bkmTpPointsFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toDouble)).toArray

    val sortedClusters = Source.fromFile(bkmTpClustersFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toInt)).toArray

    val clusterOptDistance = Source.fromFile(bkmClusterOptDistanceFile.format(numClusters, attrChangePercentage)).getLines().next()
      .split(",", -1).map(_.toDouble)
      
    val successCount = launchEvasionAttack(tpPoints, sortedClusters, bkmModel.clusterCenters, clusterInfoArray, clusterOptDistance, attrChangePercentage)
    
    println("BKMeans evasion success at %.2f: %d".format(attrChangePercentage, successCount))
    
    (attrChangePercentage, successCount)
  }
  
  def gmQuickEvasionAttack(sc: SparkContext, numClusters: Int, attrChangePercentage: Double): (Double, Int) = {

    val gmModel = GaussianMixtureModel.load(sc, gmModelDir.format(numClusters))

    val clusterInfoArray: Array[ClusterInfo] = AnomalyDetection.getClusterInfoArray(sc, gmModelClusterFile.format(numClusters))

    val tpPoints = Source.fromFile(gmTpPointsFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toDouble)).toArray

    val sortedClusters = Source.fromFile(gmTpClustersFile.format(numClusters)).getLines()
      .map(_.split(",", -1).map(_.toInt)).toArray

    val clusterOptDistance = Source.fromFile(gmClusterOptDistanceFile.format(numClusters, attrChangePercentage)).getLines().next()
      .split(",", -1).map(_.toDouble)
      
    val successCount = launchEvasionAttack(tpPoints, sortedClusters, gmModel.gaussians.map(_.mu), clusterInfoArray, clusterOptDistance, attrChangePercentage)
    
    println("GM evasion success at %.2f: %d".format(attrChangePercentage, successCount))
    
    (attrChangePercentage, successCount)
  }

  def launchEvasionAttack(tpPoints: Array[Array[Double]], sortedClusters: Array[Array[Int]],
        centers: Array[Vector], clusterInfoArray: Array[ClusterInfo],
        clusterOptDistance: Array[Double], attrChangePercentage: Double) : Int = {
    
    val attrChangeThresholds = AttrChangability.getAttrChangeThresholds(attrChangePercentage)
    
    var successCount = 0
    
    for (i <- 0 to tpPoints.size - 1) {

      var cplexSolution = new CplexSolution()

      for {
        j <- 0 to sortedClusters(i).size - 1
        if !cplexSolution.found
      } {

        val origX = tpPoints(i)
        val projX = AdvUtil.getProjFromOrig(Vectors.dense(tpPoints(i)), pcMatrix).toArray
        val clusterIndex = sortedClusters(i)(j)
        val projCenter = centers(clusterIndex).toArray

        val distanceFromCluster = Vectors.sqdist(Vectors.dense(projX), Vectors.dense(projCenter))

        if (distanceFromCluster <= clusterOptDistance(clusterIndex)) {

          val sqDistThres = AdvUtil.chebyshevThreshold(clusterInfoArray(clusterIndex).meanSqDist, clusterInfoArray(clusterIndex).stdDevSqDist)
          cplexSolution = CplexOptimizer.softOptimize(origX, projX, projCenter, sqDistThres, pcArray, attrChangeThresholds)

          if (cplexSolution.found){
            successCount += 1
            
            println("Evasion at %.2f: try = %d, success = %d".format(attrChangePercentage, i, successCount))
            
          }
        }
      }
    }
    
    successCount
    
  }

}
  