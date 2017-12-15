package com.icanindya.adversarial.iscx

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import java.io.PrintWriter
import java.io.FileWriter
import com.icanindya.adversarial.Optimizer
import com.icanindya.adversarial.Point
import com.icanindya.adversarial.Spark

object AttackAnomalyDetection {

  val finalDsFile = FinalDataset.joinedDsFile
  val modelDir = AnomalyDetection.modelDir
  val pcFile = AnomalyDetection.pcFile
  val FINAL_RESULT_FILE = "D:/Data/ISCX IDS/result/final_result.txt"

  var model:KMeansModel = null
  var pc:DenseMatrix = null
  var optimizer:Optimizer = null
  var pcArr:Array[Array[Double]] = null
  var projCentersArr:Array[Array[Double]] = null
  var attrChangeThrs:Array[Double] = null
  var consThresholds:Array[Double] = null

  val STD_ATTR_CHANGE_THRES = 0.2
  var objective = Double.MinValue
  val w1 = 0.3
  val w2 = 0.7
  val attackCodeToName: Map[Double, String] = Map(0.0 -> "Normal", 1.0 -> "Attack")

  val pw = new PrintWriter(new FileWriter(FINAL_RESULT_FILE), false)

  def main(args: Array[String]) {
    
    val sc = Spark.getContext()
    
//    val lines = sc.textFile(FinalDataset.scaledTrainFile).union(sc.textFile(FinalDataset.scaledTestFile)).cache()
//    for(i <- 0 to FinalDataset.NUM_FEATURES - 1){
//      val values = lines.map(line => line.split(",", -1)(i).toDouble)
//      println("i:%d %f, %f".format(i, values.min, values.max))  
//    }
    
    val labeledPointTestRdd = sc.textFile(FinalDataset.finalTestFile).map{line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }
    labeledPointTestRdd.cache()
    
//    check(sc, labeledPointTestRdd)
    getResultsForDiffDistThres(sc, labeledPointTestRdd)

    pw.close()
  }
  
  def check(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint]){
    
    val NUM_FEATURES = labeledPointTestRdd.first().features.size
    
    model = KMeansModel.load(sc, modelDir.format(100))
    pc = sc.objectFile[DenseMatrix](pcFile.format(100)).collect()(0)
    
    pcArr = pcToArr(pc)
    
    val projCenters = model.clusterCenters
    projCentersArr = model.clusterCenters.map(_.toArray)
    
    attrChangeThrs = Array.fill[Double](NUM_FEATURES)(STD_ATTR_CHANGE_THRES)
    val sqDistThres = 0.01
    var targetCluster = 68
    
    optimizer = new Optimizer(pcArr, projCentersArr, Math.sqrt(sqDistThres), attrChangeThrs, targetCluster)
        
//    val clusMemberCount  = labeledPointTestRdd.map{ testPoint =>
//      val projTestPointVec = getProjFromOrig(testPoint.features, pc)
//      val nearestClusterIndex = model.predict(projTestPointVec)
//      val sqDist = Vectors.sqdist(projTestPointVec, projCenters(nearestClusterIndex))
//      (nearestClusterIndex, 1)
//    }
//    .reduceByKey(_+_)
//    .collect.sortWith((x,y) => x._2 > y._2)
//    
//    println(clusMemberCount.mkString("\n"))
    
    var successCount = 0
    
    for(targetClsuter <- 0 to model.clusterCenters.length){
      
      val targetPoints = labeledPointTestRdd.filter { testPoint => 
        val projTestPointVec = getProjFromOrig(testPoint.features, pc)
        val nearestClusterIndex = model.predict(projTestPointVec)
        val sqDist = Vectors.sqdist(projTestPointVec, projCenters(nearestClusterIndex))
        nearestClusterIndex == targetCluster && sqDist > sqDistThres
      }
      .map{ testPoint =>
        val projTestPointVec = getProjFromOrig(testPoint.features, pc)
        val sqDist = Vectors.sqdist(projTestPointVec, projCenters(targetCluster))
        (testPoint, sqDist)      
      }
      .collect()
      .sortWith((x,y) => x._2 < y._2)
      .map{x =>
        println("dist: " + x._2)
        x._1
      }
      
      println("Target points:" + targetPoints.length)
      
      for(i <- 0 to targetPoints.length - 1){
        val projTargetPoint = getProjFromOrig(targetPoints(i).features, pc)
        println(model.clusterCenters(targetCluster).toArray.mkString(","))
        println(projTargetPoint.toArray.mkString(","))
        if(optimizer.optimize(targetPoints(i).features.toArray, projTargetPoint.toArray, model.clusterCenters(targetCluster).toArray, attrChangeThrs)){
          successCount += 1
//          println("success")
        }
        else{
//          println("fail")
        }
      }
    }
    
    println("success count: " + successCount)
  }
  

  def getResultsForDiffDistThres(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint]) {
    
    val NUM_FEATURES = labeledPointTestRdd.first().features.size
    
    for(k <- 80 to 150 by 10){
      model = KMeansModel.load(sc, modelDir.format(k))
      pc = sc.objectFile[DenseMatrix](pcFile.format(k)).collect()(0)

      pcArr = pcToArr(pc)

      val projCenters = model.clusterCenters
      projCentersArr = model.clusterCenters.map(_.toArray)

      attrChangeThrs = Array.fill[Double](NUM_FEATURES)(STD_ATTR_CHANGE_THRES)
      println("Feature Modification Threshold: %s".format(attrChangeThrs.mkString(",")))

      var minSqDistInModel = Double.MaxValue
      var maxSqDistInModel = Double.MinValue

      minSqDistInModel = 4.5
      maxSqDistInModel = 5.5

      val optSqDist = ternarySearch(minSqDistInModel, maxSqDistInModel, labeledPointTestRdd, k)
      println("k = %d, t = %f".format(k, optSqDist))
    }

  }

  def ternarySearch(leftSqDist: Double, rightSqDist: Double, labeledPointTestRdd: RDD[LabeledPoint], k: Int): Double = {
    if ((rightSqDist - leftSqDist) < 0.005) return (rightSqDist - leftSqDist) / 2
    else {
      val leftQSqDist = leftSqDist + (rightSqDist - leftSqDist) * 0.25
      val rightQSqDist = leftSqDist + (rightSqDist - leftSqDist) * 0.75
      val midSqDist = leftSqDist + (rightSqDist - leftSqDist) * 0.5

      val leftQResults = getObjectiveValue(labeledPointTestRdd, leftQSqDist, k)
      val rightQResults = getObjectiveValue(labeledPointTestRdd, rightQSqDist, k)
      val midResults = getObjectiveValue(labeledPointTestRdd, midSqDist, k)

      var nwLeftSqDist, nwRightSqDist = -1.0

      if (midResults._1 < leftQResults._1 && midResults._1 < rightQResults._1) {
        nwLeftSqDist = leftQSqDist
        nwRightSqDist = rightQSqDist
        return ternarySearch(nwLeftSqDist, nwRightSqDist, labeledPointTestRdd, k)
      } else if (midResults._1 < leftQResults._1 && midResults._1 > rightQResults._1) {
        nwLeftSqDist = midSqDist
        nwRightSqDist = rightSqDist
        return ternarySearch(nwLeftSqDist, nwRightSqDist, labeledPointTestRdd, k)
      } else { // if(midResults._1 > leftQResults._1 && midResults._1 < rightQResults._1){
        nwLeftSqDist = leftSqDist
        nwRightSqDist = midSqDist
        return ternarySearch(nwLeftSqDist, nwRightSqDist, labeledPointTestRdd, k)
      }
    }

  }

  def getObjectiveValue(labeledPointTestRdd: RDD[LabeledPoint], sqDistThres: Double, k: Int): (Double, Long, Long, Long) = {
    var TP = 0L
    var FP = 0L
    var FN = 0L
    var TN = 0L
    var MIMIC = 0L;

    val testPointInfoRdd = labeledPointTestRdd.map { testPoint =>
      var cat = ""
      val projPointVec = getProjFromOrig(testPoint.features, pc)

      val clusterIndex = model.predict(projPointVec)
      val sqDist = Vectors.sqdist(projPointVec, model.clusterCenters(clusterIndex))

      if (sqDist < sqDistThres) {
        if (testPoint.label == 0.0) {
          TN += 1
          cat = "TN"
        } else {
          FN += 1
          cat = "FN"
        }
      } else {
        if (testPoint.label != 0.0) {
          TP += 1
          cat = "TP"
        } else {
          FP += 1
          cat = "FP"
        }
      }

      new Point(testPoint.features, attackCodeToName(testPoint.label), clusterIndex, sqDist, cat, pc)
    }
    
    testPointInfoRdd.cache()

    //    print("".format(testPointInfoRdd.count()))

    TP = testPointInfoRdd.filter(x => x.classifiedAs == "TP").count()
    FP = testPointInfoRdd.filter(x => x.classifiedAs == "FP").count()
    TN = testPointInfoRdd.filter(x => x.classifiedAs == "TN").count()
    FN = testPointInfoRdd.filter(x => x.classifiedAs == "FN").count()

    val consStart = System.currentTimeMillis()

    val consThresholds = (for (i <- 0 to k - 1) yield {
      optimizer = new Optimizer(pcArr, projCentersArr, Math.sqrt(sqDistThres), attrChangeThrs, i)

      val sortedPoints = testPointInfoRdd.filter(x => x.classifiedAs == "TP").map { x =>
        val sqDist = Vectors.sqdist(x.projFeatures, model.clusterCenters(i))
        (x, sqDist)
      }.collect().sortWith(_._2 < _._2)

      binarySearch(0, sortedPoints.length - 1, sortedPoints)
    }).toArray

    val consEnd = System.currentTimeMillis()

    println("K = %d, SqDist = %f: consideration thresholds determination time = %f".format(k, sqDistThres, (consEnd - consStart) / (60 * 1000.0)))

    val mimicStart = System.currentTimeMillis()

    testPointInfoRdd.filter { x => x.classifiedAs == "TP" }.collect.foreach { x =>

      var solFound = false
      var solIndex = -1

      for {
        i <- 0 to k - 1
        if (solFound == false)
      } {
        val sqDist = Vectors.sqdist(x.projFeatures, model.clusterCenters(i))
        if (sqDist < consThresholds(i)) {
          if (optimizer.optimize(x.features.toArray, x.projFeatures.toArray, projCentersArr(i), attrChangeThrs)) {
            solFound = true
            solIndex = i
            MIMIC += 1
          }
        }
      }
    }

    val mimicEnd = System.currentTimeMillis()

    println("K = %d, SqDist = %f: mimicry attack launching time = %f".format(k, sqDistThres, (mimicEnd - mimicStart) / (60 * 1000.0)))

    var objectiveValue = w1 * FP + w2 * (FN + MIMIC)
    //    objectiveValue = (TP + TN) / (TP + TN + FP + FN).toFloat

    println("%d, %f, %f, %d, %d, %d".format(k, sqDistThres, objectiveValue, FP, FN, MIMIC))
    pw.println("%d, %f, %f, %d, %d, %d".format(k, sqDistThres, objectiveValue, FP, FN, MIMIC))
    pw.flush()
    (objectiveValue, FP, FN, MIMIC)
  }
  

  def binarySearch(left: Int, right: Int, sortedPoints: Array[(Point, Double)]): Double = {
    var l = left
    var r = right

    var s = -1

    while (l <= r) {
      //            println("l: %d, r: %d".format(l, r))

      val m = Math.floor((l + r) / 2.0).toInt
      optimizer.setClosestClusterInfo(sortedPoints(m)._1.nearestCluster, sortedPoints(m)._1.nearestSqDist)
      if (optimizer.optimize(sortedPoints(m)._1.features.toArray, sortedPoints(m)._1.projFeatures.toArray)) {
        s = m
        l = m + 1
      } else r = m - 1
    }
    return if (s == -1) -1 else sortedPoints(s)._2
  }

  def getProjFromOrig(vector: Vector, pc: DenseMatrix): Vector = {
    var pointMat = new DenseMatrix(1, vector.size, vector.toArray)
    var projPoint = pointMat.multiply(pc);
    new DenseVector(projPoint.toArray);
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
}