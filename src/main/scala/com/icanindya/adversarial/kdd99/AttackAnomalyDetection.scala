//package com.icanindya.adversarial.kdd99
//
//import org.apache.spark.SparkContext
//import org.apache.spark.SparkContext._
//import org.apache.spark.mllib.clustering.KMeansModel
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.rdd._
//import org.apache.spark.mllib.linalg._
//import org.apache.spark.mllib.regression.LabeledPoint
//import java.io.PrintWriter
//import java.io.FileWriter
//import com.icanindya.adversarial.Optimizer
//import com.icanindya.adversarial.Point
//import com.icanindya.adversarial.Spark
//import com.icanindya.adversarial.AdvUtil
//
//object AttackAnomalyDetection {
//
//  var model: KMeansModel = null
//  var pc: DenseMatrix = null
//  var optimizer: Optimizer = null
//  var pcArr: Array[Array[Double]] = null
//  var projCentersArr: Array[Array[Double]] = null
//  var attrChangeThrs: Array[Double] = null
//  var consThresholds: Array[Double] = null
//
//  val STD_ATTR_CHANGE_THRES = 0.2
//  var objective = Double.MinValue
//  val w1 = 0.3
//  val w2 = 0.7
//  val attackCodeToName: Map[Double, String] = Map(0.0 -> "Normal", 1.0 -> "Attack")
//
//  val FINAL_RESULT_FILE = "D:/Data/KDD99/results/mimicry_results.txt"
//  val pw = new PrintWriter(new FileWriter(FINAL_RESULT_FILE), false)
//
//  def main(args: Array[String]): Unit = {
//
//    val sc = Spark.getContext()
//
//    val attackPointsInTrain = sc.textFile(FinalDataset.finalTrainFile).map { line =>
//      val values = line.split(",", -1).map(_.toDouble)
//      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
//    }
//      .filter(_.label == 1.0)
//
//    val allPointsInTest = sc.textFile(FinalDataset.finalTrainFile).map { line =>
//      val values = line.split(",", -1).map(_.toDouble)
//      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
//    }
//
//    val labeledPointTestRdd = attackPointsInTrain.union(allPointsInTest)
//    labeledPointTestRdd.cache()
//    
//    attrChangeThrs = AttrChangeThreshold.getArray()
//
//    for (k <- 80 to 150 by 10) {
//
//      var TP = 0L
//      var FP = 0L
//      var FN = 0L
//      var TN = 0L
//
//      model = KMeansModel.load(sc, AnomalyDetection.kmModelDir.format(k))
//      pc = sc.objectFile[DenseMatrix](AnomalyDetection.pcFile.format(k)).collect()(0)
//      
//      pcArr = pcToArr(pc)
//    
//      val projCenters = model.clusterCenters
//      projCentersArr = model.clusterCenters.map(_.toArray)
//      
//      
//      var minSqDist = Double.MaxValue
//      var maxSqDist = Double.MinValue
//      
//      println("feature size: " + labeledPointTestRdd.first.features.size)
//
//      val testPointInfoRdd = labeledPointTestRdd.collect.foreach { lp =>
//        var cat = ""
//        val projPointVec = AdvUtil.getProjFromOrig(lp.features, pc)
//
//        val clusterIndex = model.predict(projPointVec)
//        val sqDist = Vectors.sqdist(projPointVec, model.clusterCenters(clusterIndex))
//
//        if (sqDist < minSqDist) minSqDist = sqDist
//        else if (sqDist > maxSqDist) maxSqDist = sqDist
//      }
//
//      for (i <- 0 to 10) {
//        val sqDistThres = minSqDist + i * (maxSqDist - minSqDist) / 10
//
//        getObjectiveValue(labeledPointTestRdd, sqDistThres, k)
//
//      }
//
//    }
//
//  }
//
//  def getObjectiveValue(labeledPointTestRdd: RDD[LabeledPoint], sqDistThres: Double, k: Int) {
//    var TP = 0L
//    var FP = 0L
//    var FN = 0L
//    var TN = 0L
//    var MIMIC = 0L;
//
//    val testPointInfoRdd = labeledPointTestRdd.map { testPoint =>
//      var cat = ""
//      val projPointVec = getProjFromOrig(testPoint.features, pc)
//
//      val clusterIndex = model.predict(projPointVec)
//      val sqDist = Vectors.sqdist(projPointVec, model.clusterCenters(clusterIndex))
//
//      if (sqDist < sqDistThres) {
//        if (testPoint.label == 0.0) {
//          TN += 1
//          cat = "TN"
//        } else {
//          FN += 1
//          cat = "FN"
//        }
//      } else {
//        if (testPoint.label != 0.0) {
//          TP += 1
//          cat = "TP"
//        } else {
//          FP += 1
//          cat = "FP"
//        }
//      }
//
//      new Point(testPoint.features, attackCodeToName(testPoint.label), clusterIndex, sqDist, cat, pc)
//    }
//
//    testPointInfoRdd.cache()
//
//    TP = testPointInfoRdd.filter(x => x.classifiedAs == "TP").count()
//    FP = testPointInfoRdd.filter(x => x.classifiedAs == "FP").count()
//    TN = testPointInfoRdd.filter(x => x.classifiedAs == "TN").count()
//    FN = testPointInfoRdd.filter(x => x.classifiedAs == "FN").count()
//
//    val consStart = System.currentTimeMillis()
//
//    val consThresholds = (for (i <- 0 to k - 1) yield {
//      optimizer = new Optimizer(pcArr, projCentersArr, Math.sqrt(sqDistThres), attrChangeThrs, i)
//
//      val sortedPoints = testPointInfoRdd.filter(x => x.classifiedAs == "TP").map { x =>
//        val sqDist = Vectors.sqdist(x.projFeatures, model.clusterCenters(i))
//        (x, sqDist)
//      }.collect().sortWith(_._2 < _._2)
//
//      binarySearch(0, sortedPoints.length - 1, sortedPoints)
//    }).toArray
//
//    val consEnd = System.currentTimeMillis()
//
//    println("K = %d, SqDist = %f: consideration thresholds determination time = %f".format(k, sqDistThres, (consEnd - consStart) / (60 * 1000.0)))
//
//    val mimicStart = System.currentTimeMillis()
//
//    testPointInfoRdd.filter { x => x.classifiedAs == "TP" }.collect.foreach { x =>
//
//      var solFound = false
//      var solIndex = -1
//
//      for {
//        i <- 0 to k - 1
//        if (solFound == false)
//      } {
//        val sqDist = Vectors.sqdist(x.projFeatures, model.clusterCenters(i))
//        if (sqDist < consThresholds(i)) {
//          if (optimizer.optimize(x.features.toArray, x.projFeatures.toArray, projCentersArr(i), attrChangeThrs)) {
//            solFound = true
//            solIndex = i
//            MIMIC += 1
//          }
//        }
//      }
//    }
//
//    val mimicEnd = System.currentTimeMillis()
//
//    println("K = %d, SqDist = %f: mimicry attack launching time = %f".format(k, sqDistThres, (mimicEnd - mimicStart) / (60 * 1000.0)))
//
//    var objectiveValue = w1 * FP + w2 * (FN + MIMIC)
//
//    println("%d, %f, %d, %d, %d, %f".format(k, sqDistThres, FP, FN, MIMIC, objectiveValue))
//    pw.println("%d, %f, %d, %d, %d, %f".format(k, sqDistThres, FP, FN, MIMIC, objectiveValue))
//    pw.flush()
//    
//  }
//
//  def binarySearch(left: Int, right: Int, sortedPoints: Array[(Point, Double)]): Double = {
//    var l = left
//    var r = right
//
//    var s = -1
//
//    while (l <= r) {
//      val m = Math.floor((l + r) / 2.0).toInt
//      optimizer.setClosestClusterInfo(sortedPoints(m)._1.nearestCluster, sortedPoints(m)._1.nearestSqDist)
//      if (optimizer.optimize(sortedPoints(m)._1.features.toArray, sortedPoints(m)._1.projFeatures.toArray)) {
//        s = m
//        l = m + 1
//      } else r = m - 1
//    }
//    return if (s == -1) -1 else sortedPoints(s)._2
//  }
//
//  def getProjFromOrig(vector: Vector, pc: DenseMatrix): Vector = {
//    var pointMat = new DenseMatrix(1, vector.size, vector.toArray)   
//    var projPoint = pointMat.multiply(pc);
//    new DenseVector(projPoint.toArray);
//  }
//
//  def pcToArr(pc: Matrix): Array[Array[Double]] = {
//    var c = 0
//    val pcArr = Array.ofDim[Double](pc.numRows, pc.numCols)
//
//    //pc.toArray is a column major array
//    for (i <- 0 to pc.numCols - 1) {
//      for (j <- 0 to pc.numRows - 1) {
//        pcArr(j)(i) = pc.toArray(c)
//        c += 1
//      }
//    }
//    pcArr
//  }
//
//}