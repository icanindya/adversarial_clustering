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
//import com.icanindya.adversarial.kdd99.AttrChangeThreshold
//import com.icanindya.adversarial.Optimizer
//import com.icanindya.adversarial.Point
//import com.icanindya.adversarial.Spark
//
//object AttackApproach2Looping {
//
//  val trainDataFile = ""
//  val testDataFile = ""
//  val modelsDir = "D:/Data/KDD99/Metadata/Approach2New/Models/%d"
//  val pcsFile = "D:/Data/KDD99/Metadata/Approach2New/PCs/%d"
//  val FINAL_RESULT_FILE = "D:/Data/KDD99/final_result.txt"
//
//  var attackNameToCode: Map[String, Int] = Map()
//  var attackCodeToName: Map[Int, String] = Map()
//  var protocols: Map[String, Int] = Map()
//  var services: Map[String, Int] = Map()
//  var flags: Map[String, Int] = Map()
//
//  var model = new KMeansModel(Array.ofDim[Vector](0))
//  var pc: DenseMatrix = null
//  var optimizer = new Optimizer(null, null, 0, null, -1)
//  var pcArr = Array.ofDim[Array[Double]](0)
//  var projCentersArr = Array.ofDim[Array[Double]](0)
//  var attrChangeThrs = Array.ofDim[Double](0)
//
//  val STD_ATTR_CHANGE_THRES = 0.2
//  var objective = Double.MinValue
//  val w1 = 0.3
//  val w2 = 0.7
//
//  var consThresholds = Array[Double]()
//
//  val pw = new PrintWriter(new FileWriter(FINAL_RESULT_FILE), false)
//
//  def main(args: Array[String]) {
//    val sc = Spark.getContext()
//
//    val trainData = sc.textFile(trainDataFile).cache()
//    val testData = sc.textFile(testDataFile).distinct().cache()
//    val data = trainData.union(testData).cache()
//
//    attackNameToCode = data.map(_.split(',')(41).dropRight(1)).distinct().collect().zipWithIndex.toMap
//    attackCodeToName = attackNameToCode.map(_.swap)
//    protocols = data.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
//    services = data.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
//    flags = data.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap
//
//        getResultsForDiffDistThres(sc, testData)
////    getAnomalyResultsForDiffDistThres(sc, testData)
//
//    pw.close()
//  }
//
//  def getAnomalyResultsForDiffDistThres(sc: SparkContext, testData: RDD[String]) {
//    val labeledPointTestRdd = processData(testData).cache()
//    for {
//      k <- 10 to 120
//      if k % 10 == 0
//    } {
//
//      println("k = " + k)
//      var TP = 0L
//      var FP = 0L
//      var FN = 0L
//      var TN = 0L
//
//      model = KMeansModel.load(sc, modelsDir.format(k.toString()))
//      pc = sc.objectFile[DenseMatrix](pcsFile.format(k.toString())).collect()(0)
//      
//      pcArr = pcToArr(pc)
//
//      projCentersArr = model.clusterCenters.map(_.toArray)
//
//      val testPointInfoRdd = labeledPointTestRdd.foreach { lp =>
//        var cat = ""
//        val projPointVec = getProjFromOrig(lp.features, pc)
//
//        val clusterIndex = model.predict(projPointVec)
//        val sqDist = Vectors.sqdist(projPointVec, model.clusterCenters(clusterIndex))
//
//        if (sqDist < 0.3) {
//          if (lp.label == attackNameToCode("normal")) {
//            TN += 1
//            cat = "TN"
//          } else {
//            FN += 1
//            cat = "FN"
//          }
//        } else {
//          if (lp.label != attackNameToCode("normal")) {
//            TP += 1
//            cat = "TP"
//          } else {
//            FP += 1
//            cat = "FP"
//          }
//        }
//      }
//
//      println("precision = " + TP / (TP + FP).toFloat)
//      println("recall = " + TP / (TP + FN).toFloat)
//      println("Accuracy = " + (TP + TN) / (TP + TN + FP + FN).toFloat)
//
//    }
//  }
//
//  def getResultsForDiffDistThres(sc: SparkContext, testData: RDD[String]) {
//    val labeledPointTestRdd = processData(testData)
//    labeledPointTestRdd.cache()
//    
//    for {
//      k <- 80 to 120
//      if k % 10 == 0
//    } {
//
//      model = KMeansModel.load(sc, modelsDir.format(k))
//      pc = sc.objectFile[DenseMatrix](pcsFile.format(k)).collect()(0)
//
//      pcArr = pcToArr(pc)
//
//      val projCenters = model.clusterCenters
//      projCentersArr = model.clusterCenters.map(_.toArray)
//
//      attrChangeThrs = AttrChangeThreshold.getArray().map { x => if (x == 0.0) 0.0 else STD_ATTR_CHANGE_THRES }
//      println("Feature Modification Threshold: %s".format(attrChangeThrs.mkString(",")))
//
//      var minSqDistInModel = Double.MaxValue
//      var maxSqDistInModel = Double.MinValue
//
//      minSqDistInModel = 0
//      maxSqDistInModel = 1.5
//
//      val optSqDist = ternarySearch(minSqDistInModel, maxSqDistInModel, labeledPointTestRdd, k)
//      println("k = %d, t = %f".format(k, optSqDist))
//    }
//
//  }
//
//  def ternarySearch(leftSqDist: Double, rightSqDist: Double, labeledPointTestRdd: RDD[LabeledPoint], k: Int): Double = {
//    if ((rightSqDist - leftSqDist) < 0.1) return (rightSqDist - leftSqDist) / 2
//    else {
//      val leftQSqDist = leftSqDist + (rightSqDist - leftSqDist) * 0.25
//      val rightQSqDist = leftSqDist + (rightSqDist - leftSqDist) * 0.75
//      val midSqDist = leftSqDist + (rightSqDist - leftSqDist) * 0.5
//
//      val leftQResults = getObjectiveValue(labeledPointTestRdd, leftQSqDist, k)
//      val rightQResults = getObjectiveValue(labeledPointTestRdd, rightQSqDist, k)
//      val midResults = getObjectiveValue(labeledPointTestRdd, midSqDist, k)
//
//      var nwLeftSqDist, nwRightSqDist = -1.0
//
//      if (midResults._1 < leftQResults._1 && midResults._1 < rightQResults._1) {
//        nwLeftSqDist = leftQSqDist
//        nwRightSqDist = rightQSqDist
//        return ternarySearch(nwLeftSqDist, nwRightSqDist, labeledPointTestRdd, k)
//      } else if (midResults._1 < leftQResults._1 && midResults._1 > rightQResults._1) {
//        nwLeftSqDist = midSqDist
//        nwRightSqDist = rightSqDist
//        return ternarySearch(nwLeftSqDist, nwRightSqDist, labeledPointTestRdd, k)
//      } else { // if(midResults._1 > leftQResults._1 && midResults._1 < rightQResults._1){
//        nwLeftSqDist = leftSqDist
//        nwRightSqDist = midSqDist
//        return ternarySearch(nwLeftSqDist, nwRightSqDist, labeledPointTestRdd, k)
//      }
//    }
//
//  }
//
//  def processData(data: RDD[String]): RDD[LabeledPoint] = {
//    data.map { line =>
//
//      val buffer = line.split(",").toBuffer
//      val protocol = buffer.remove(1)
//      val service = buffer.remove(1)
//      val tcpState = buffer.remove(1)
//      val label = attackNameToCode(buffer.remove(buffer.length - 1).dropRight(1))
//      val vector = buffer.map(_.toDouble)
//
//      val newProtocolFeatures = new Array[Double](protocols.size)
//      newProtocolFeatures(protocols(protocol)) = 1.0
//      val newServiceFeatures = new Array[Double](services.size)
//      newServiceFeatures(services(service)) = 1.0
//      val newTcpStateFeatures = new Array[Double](flags.size)
//      newTcpStateFeatures(flags(tcpState)) = 1.0
//
//      vector.insertAll(1, newTcpStateFeatures)
//      vector.insertAll(1, newServiceFeatures)
//      vector.insertAll(1, newProtocolFeatures)
//
//      new LabeledPoint(label, Vectors.dense(vector.toArray))
//    }
//  }
//
//  def binarySearch(left: Int, right: Int, sortedPoints: Array[(Point, Double)]): Double = {
//    var l = left
//    var r = right
//
//    var s = -1
//
//    while (l <= r) {
//      //            println("l: %d, r: %d".format(l, r))
//
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
//  def getObjectiveValue(labeledPointTestRdd: RDD[LabeledPoint], sqDistThres: Double, k: Int): (Double, Long, Long, Long) = {
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
//        if (testPoint.label == attackNameToCode("normal")) {
//          TN += 1
//          cat = "TN"
//        } else {
//          FN += 1
//          cat = "FN"
//        }
//      } else {
//        if (testPoint.label != attackNameToCode("normal")) {
//          TP += 1
//          cat = "TP"
//        } else {
//          FP += 1
//          cat = "FP"
//        }
//      }
//
//      new Point(testPoint.features, attackCodeToName(testPoint.label.toInt), clusterIndex, sqDist, cat, pc)
//    }
//    
//    testPointInfoRdd.cache()
//
//    //    print("".format(testPointInfoRdd.count()))
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
//          println(x.projFeatures.toArray.mkString(","))
//          println(projCentersArr(i).mkString(","))
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
//    //    objectiveValue = (TP + TN) / (TP + TN + FP + FN).toFloat
//
//    println("%d, %f, %f, %d, %d, %d".format(k, sqDistThres, objectiveValue, FP, FN, MIMIC))
//    pw.println("%d, %f, %f, %d, %d, %d".format(k, sqDistThres, objectiveValue, FP, FN, MIMIC))
//    pw.flush()
//    (objectiveValue, FP, FN, MIMIC)
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
//}