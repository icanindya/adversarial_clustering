package com.icanindya.adversarial.kdd99

import com.icanindya.adversarial._
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.DenseMatrix
import com.icanindya.adversarial.iscx.DrMuratAnomalyDetection


object DrMuratAdversarialAttack {

  val optMap = Map(5 -> (10, 1.2),
    10 -> (10, 1.1),
    15 -> (10, 1.1),
    20 -> (10, 1.1),
    25 -> (10, 1.1))
    
    
  val fAttacks = Array(0.0, 0.3, 0.5, 0.7, 1.0)

  def main(args: Array[String]): Unit = {

    val sc = Spark.getContext()

    val labeledPointValidRdd = sc.textFile(FinalDataset.finalValidationFile).map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }
    labeledPointValidRdd.cache

    var labeledPointTestRdd = sc.textFile(FinalDataset.finalTestFile).map { line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }
    labeledPointTestRdd.cache()

    val benLpTestRdd = labeledPointTestRdd.filter(x => x.label == 0.0)
    benLpTestRdd.cache()

    val malLpTestRdd = labeledPointTestRdd.filter(x => x.label == 1.0).sample(false, (benLpTestRdd.count * 1.1) / labeledPointTestRdd.count, 123456)

    labeledPointTestRdd = benLpTestRdd.union(malLpTestRdd)
    labeledPointTestRdd.cache()

    val pcMatrix = sc.objectFile[DenseMatrix](DrMuratAnomalyModel.pcFile.format(DrMuratAnomalyModel.NUM_PC)).collect()(0)

    for (e <- 5 to 25 by 5) {

      var (tp, tn, fp, fn) = (0, 0, 0, 0)

      val k = optMap(e)._1
      val sqDistThres = Math.pow(optMap(e)._2, 2)
      val kmModel = KMeansModel.load(sc, DrMuratAnomalyModel.kmModelDir.format(k))
      val mimicryTargetPoints = labeledPointValidRdd.filter { validPoint =>
        val projTestPoint = AdvUtil.getProjFromOrig(validPoint.features, pcMatrix)
        val label = validPoint.label

        val clusterIndex = kmModel.predict(projTestPoint)
        val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))

        sqDist < sqDistThres
      }
      .map(_.features)
      .takeSample(false, 100, 123456)

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

      val tpTestPointRdd = labeledPointTestRdd.filter { testPoint =>
        val projTestPoint = AdvUtil.getProjFromOrig(testPoint.features, pcMatrix)
        val label = testPoint.label

        val clusterIndex = kmModel.predict(projTestPoint)
        val sqDist = Vectors.sqdist(projTestPoint, kmModel.clusterCenters(clusterIndex))

        sqDist > sqDistThres
      }
      .map(_.features)
      
      tpTestPointRdd.map{tpPoint =>
        val tpPointArr = tpPoint.toArray
        val mimicryTargetPointArr = mimicryTargetPoints.map(x => (x, Vectors.sqdist(x, tpPoint))).sortWith(_._2 < _._2)(0)._1.toArray
        
        val attackPoints = fAttacks.map{fAttack => 
          val attackPoint = Vectors.dense(tpPointArr.zip(mimicryTargetPointArr).map(x => x._1 + fAttack * (x._2 - x._1)))
          
          val projAttackPoint = AdvUtil.getProjFromOrig(attackPoint, pcMatrix)
          val clusterIndex = kmModel.predict(projAttackPoint)
          
          val sqDist = Vectors.sqdist(projAttackPoint, kmModel.clusterCenters(clusterIndex))
          
          if(sqDist < sqDistThres) 1
          
        }
        
      }

    }

  }
}