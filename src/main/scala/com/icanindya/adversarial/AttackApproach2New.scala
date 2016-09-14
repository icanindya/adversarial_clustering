package com.icanindya.adversarial

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.mllib.clustering.KMeansModel
import java.io.FileWriter
import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer
import org.apache.parquet.filter2.compat.FilterCompat.Filter

object Attack {

  val trainDataFile = Constants.trainFile
  val testDataFile = Constants.testFile
  val modelDir = "E:/Data/KDD99/Metadata/Approach2New/Model"
  val pcFile = "E:/Data/KDD99/Metadata/Approach2New/pc"
  val pcPrintFile = "E:/Data/KDD99/Metadata/Approach2New/pc_print"
  val typeStatFile = "E:/Data/KDD99/Approach2New/type_stat"

  var attackNameToCode: Map[String, Int] = Map()
  var attackCodeToName: Map[Int, String] = Map()
  var protocols: Map[String, Int] = Map()
  var services: Map[String, Int] = Map()
  var flags: Map[String, Int] = Map()
  var sqDistThres = Constants.sqDistThres
  var TP = 0L
  var FP = 0L
  var FN = 0L
  var TN = 0L
  var succ = 0L;

  var mTry = 0L;
  var mAtt = 0L;

  type TestPointInfoType = RDD[(Vector, String, String, Double, Int)]

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir", "C:/hadoop241/");
    val conf = new SparkConf().setAppName("adversarial").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.memory", "10g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val trainData = sc.textFile(trainDataFile);
    val testData = sc.textFile(testDataFile)
    val data = trainData.union(testData)

    attackNameToCode = data.map(_.split(',')(41).dropRight(1)).distinct().collect().zipWithIndex.toMap
    attackCodeToName = attackNameToCode.map(_.swap)
    protocols = data.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    services = data.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    flags = data.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap

    println(attackCodeToName)

    val labeledPointTrainRdd = processData(trainData)

    val model = KMeansModel.load(sc, modelDir)
    val pc = sc.objectFile[Matrix](pcFile).collect()(0)

    val pcArr = pcToArr(pc)
    printPc(pcArr, pcPrintFile)

    val projCentersArr = model.clusterCenters.map(_.toArray)
    println("Feature change thresholds: %s".format(AttrChangeThreshold.getArray().mkString(",")))
    val optimizer = new Optimizer(pcArr, projCentersArr, Math.sqrt(sqDistThres), AttrChangeThreshold.getArray())

    var clusAssocInfo = Array.fill[ClusTestInfo](projCentersArr.length)(new ClusTestInfo)

    val labeledPointTestRdd = processData(testData)

    val testPointInfoRdd = labeledPointTestRdd.map { lp =>
      var cat = "tn"
      var pointMat = new DenseMatrix(1, lp.features.size, lp.features.toArray)
      var projPoint = pointMat.multiply(new DenseMatrix(pc.numRows, pc.numCols, pc.toArray));
      var projPointVec = new DenseVector(projPoint.toArray);

      val clusterIndex = model.predict(projPointVec)
      val sqDist = Vectors.sqdist(projPointVec, model.clusterCenters(clusterIndex))

      clusAssocInfo(clusterIndex).numAssocTestPoints += 1
      clusAssocInfo(clusterIndex).totSqDist += sqDist
      if (sqDist < clusAssocInfo(clusterIndex).minSqDist) clusAssocInfo(clusterIndex).minSqDist += sqDist
      if (sqDist > clusAssocInfo(clusterIndex).maxSqDist) clusAssocInfo(clusterIndex).maxSqDist += sqDist

      if (sqDist < sqDistThres) {
        if (lp.label == attackNameToCode("normal")) {
          TN += 1
          cat = "TN"
        } else {
          FN += 1
          cat = "FN"
        }
      } else {
        mTry += 1
        if (lp.label != attackNameToCode("normal")) {
          TP += 1
          cat = "TP"
          //              if(sqDist < 1.5){
          //                optimizer.setClosestClusterInfo(clusterIndex, sqDist);
          //                if(optimizer.optimize(lp.features.toArray, projPointVec.toArray)) succ += 1
          //                mAtt += 1
          //              }
        } else {
          FP += 1
          cat = "FP"
        }
      }

      (lp.features, attackCodeToName(lp.label.toInt), cat, sqDist, clusterIndex)
    }

    getClusterAssocAll(testPointInfoRdd)

    println("Successful mimicry attacks: " + succ)
    println("TP is : " + TP)
    println("FP is : " + FP)
    println("TN is : " + TN)
    println("FN is : " + FN)
    println("precision is : " + TP / (TP + FP).toFloat)
    println("recall is : " + TP / (TP + FN).toFloat)
    println("Accuracy is :" + (TP + TN) / (TP + TN + FP + FN).toFloat)

    //    for(i <- 0 to clusAssocInfo.length - 1){
    //      println("\nCluster %d:".format(i))
    //      println("Num of associated test points: %d".format(clusAssocInfo(i).numAssocTestPoints))
    //      println("Avg squared distance of associated points: %f".format(clusAssocInfo(i).totSqDist/clusAssocInfo(i).numAssocTestPoints))
    //      println("Min squared distance: %f".format(clusAssocInfo(i).minSqDist))
    //      println("Max squared distance: %f".format(clusAssocInfo(i).maxSqDist))
    //    }

    sc.stop()
  }

  def processData(data: RDD[String]): RDD[LabeledPoint] = {
    data.map { line =>

      val buffer = line.split(",").toBuffer
      val protocol = buffer.remove(1)
      val service = buffer.remove(1)
      val tcpState = buffer.remove(1)
      val label = attackNameToCode(buffer.remove(buffer.length - 1).dropRight(1))
      val vector = buffer.map(_.toDouble)

      val newProtocolFeatures = new Array[Double](protocols.size)
      newProtocolFeatures(protocols(protocol)) = 1.0
      val newServiceFeatures = new Array[Double](services.size)
      newServiceFeatures(services(service)) = 1.0
      val newTcpStateFeatures = new Array[Double](flags.size)
      newTcpStateFeatures(flags(tcpState)) = 1.0

      vector.insertAll(1, newTcpStateFeatures)
      vector.insertAll(1, newServiceFeatures)
      vector.insertAll(1, newProtocolFeatures)

      new LabeledPoint(label, Vectors.dense(vector.toArray))
    }
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

  def printPc(pcArr: Array[Array[Double]], outFile: String) {
    val fw = new FileWriter(pcPrintFile)
    val builder = new StringBuilder()
    try {
      for (i <- 0 to pcArr.length - 1) {
        fw.write(pcArr(i).mkString(","))
        fw.write("\n")
      }
    } finally {
      fw.close()
    }
  }

  def getFreqByClass(testPointInfoRdd: TestPointInfoType): Map[String, Long] = {
    testPointInfoRdd.map(_._2).countByValue().toSeq.sortBy(_._2).toMap
  }

  def getMissRateByClass(testPointInfoRdd: TestPointInfoType) {
    val origFreqByClass = getFreqByClass(testPointInfoRdd)
    println(origFreqByClass.toSeq.sortWith(_._2 > _._2))

    val missFreqByClass = getFreqByClass(testPointInfoRdd.filter(x => x._3 == "FN" || x._3 == "FP"))

    val missRateByClass = attackNameToCode.keys.map { key =>
      if (missFreqByClass.contains(key)) ((key, missFreqByClass(key) * 100.0 / origFreqByClass(key)))
      else (key, 0.0)
    }.toSeq.sortWith(_._2 > _._2)

    println(missRateByClass)
  }

  def getAvgSqDistByClass(testPointInfoRdd: TestPointInfoType) {
    val tpPointInfoRdd = testPointInfoRdd.filter(_._3 == "TP").cache()

    val tpSqDistance = tpPointInfoRdd.
      map { x => (x._2, (x._4, 1)) }.
      reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).
      map(x => (x._1, x._2._1 / x._2._2)).collect().toSeq.sortWith(_._2 > _._2)

    println(tpSqDistance)
  }

  def getCdfSqDist(testPointInfoRdd: TestPointInfoType) {
    val tpPointInfoRdd = testPointInfoRdd.filter(_._3 == "TP").cache()
    val minTpDist = tpPointInfoRdd.map(_._4).min()
    val maxTpDist = tpPointInfoRdd.map(_._4).max()

    for (dist <- minTpDist to maxTpDist by (maxTpDist - minTpDist) / 20) {
      println("%.2f,%.2f".format(dist, tpPointInfoRdd.filter(_._4 <= dist).count() * 100.0 / tpPointInfoRdd.count()))
    }
  }

  def getCdfClusterAssoc(testPointInfoRdd: TestPointInfoType) {
    var partTestPiontInfoRdd = testPointInfoRdd
    for (i <- 0 to Constants.numClusters - 1) {
      if (i % 10 == 0) partTestPiontInfoRdd = testPointInfoRdd.filter(_._5 < i + 10).cache()
      println("%d,%.2f".format(i, partTestPiontInfoRdd.filter(_._5 <= i).count() * 100.0 / testPointInfoRdd.count()))
    }

    println()
  }

  def getClusterAssocAll(testPiontInfoRdd: TestPointInfoType) {

    val benignTestPointInfoRdd = testPiontInfoRdd.filter(x => x._3 == "TN" || x._3 == "FP")
    for (i <- benignTestPointInfoRdd.map(_._5).countByValue().toSeq.sortWith(_._1 < _._1)) println("%d,%d".format(i._1, i._2))
    println("\n\n\n\n\n")

    val attackTestPointInfoRdd = testPiontInfoRdd.filter(x => x._3 == "FN" || x._3 == "TP")
    for (i <- attackTestPointInfoRdd.map(_._5).countByValue().toSeq.sortWith(_._1 < _._1)) println("%d,%d".format(i._1, i._2))
    println("\n\n\n\n\n")

    val tpPointInfoRdd = testPiontInfoRdd.filter(x => x._3 == "TP")
    for (i <- tpPointInfoRdd.map(_._5).countByValue().toSeq.sortWith(_._1 < _._1)) println("%d,%d".format(i._1, i._2))
    println("\n\n\n\n\n")

    val fpPointInfoRdd = testPiontInfoRdd.filter(x => x._3 == "FP")
    for (i <- fpPointInfoRdd.map(_._5).countByValue().toSeq.sortWith(_._1 < _._1)) println("%d,%d".format(i._1, i._2))
    println("\n\n\n\n\n")

    val tnPointInfoRdd = testPiontInfoRdd.filter(x => x._3 == "TN")
    for (i <- tnPointInfoRdd.map(_._5).countByValue().toSeq.sortWith(_._1 < _._1)) println("%d,%d".format(i._1, i._2))
    println("\n\n\n\n\n")

    val fnPointInfoRdd = testPiontInfoRdd.filter(x => x._3 == "FN")
    for (i <- fnPointInfoRdd.map(_._5).countByValue().toSeq.sortWith(_._1 < _._1)) println("%d,%d".format(i._1, i._2))
    println("\n\n\n\n\n")

  }
  
  var globalArr = Array.ofDim[(Vector, String, String, Double, Int)](0)

  def setConsiderationThreshold(testPointInfoRdd: TestPointInfoType) {
    val a = testPointInfoRdd.filter(x => x._3 == "TP").groupBy(_._5).map(x => (x._1, x._2.toSeq.sortWith(_._4 < _._4)))
    a.map{x => 
      globalArr = x._2.toArray
//      for(i <- 0 to arr.length){
//        
//      }
    }
  }
  
  def binarySearch(left: Int, right: Int): Int = {
    var l = left
    var r = right
    
    if(l > r) return -1
    val m = Math.floor((l + r)/2.0).toInt
    if(true) l = m + 1
    else r = m - 1
    binarySearch(l, r)
  }

  def getRangeByclass(labeledPointRdd: RDD[LabeledPoint]) {
    val testLabels = labeledPointRdd.map(_.label).distinct().collect()

    for (label <- testLabels) {
      val filteredPoints = labeledPointRdd.filter(_.label == label)
      val rangeForClass = for (i <- 0 to Constants.numAttr - 1) yield "%s: [%.5f-%.5f]".format(attackCodeToName(label.toInt), filteredPoints.map(_.features(i)).min(), filteredPoints.map(_.features(i)).max())
      println(rangeForClass.mkString(","))
    }
  }

}
