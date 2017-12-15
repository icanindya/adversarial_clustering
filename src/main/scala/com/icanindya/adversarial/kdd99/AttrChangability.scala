package com.icanindya.adversarial.kdd99

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import scala.io.Source
import java.io.PrintWriter
import java.io.File

object AttrChangability {

  val attributesFile = ""

  val attributesInfoFile = "D:/Data/KDD99/attribute_info.txt"
  val trainDataFile = ""
  val testDataFile = ""
  val outputFile = "D:/Data/KDD99/attribute_changability.txt"

  val NUM_ORIG_ATTR = 41

  //attributes: 1 + 3 + 71 + 11 + 37 = 123 // the 2nd, 3rd and 4th attributes are converted to 3, 71, 11 boolean attributes
  var attrChangeRanges: Array[Double] = Array.fill[Double](123)(0.0)

  def main(args: Array[String]): Unit = {
    saveAttrChangability()
  }

  def saveAttrChangability() {
    val lines = Source.fromFile(attributesInfoFile).getLines().toArray
    for (i <- 0 to NUM_ORIG_ATTR - 1) {
      val attName = lines(i).split(":")(1).trim()
      val attType = lines(i).split(":")(2).trim()
      var domain = lines(i).split(":")(3).trim()

      var changeRange = 0.0
      if (attType == "con") {
        domain = domain.substring(1, domain.length() - 2);
        val extremeVals = domain.split("-").map(_.trim().toDouble)
        changeRange = (extremeVals(1) - extremeVals(0))

        if (i == 0) attrChangeRanges(i) = changeRange
        else {
          attrChangeRanges(i + 1 + 3 + 71 + 11 - 4) = changeRange
        }
      }
    }

    val pw = new PrintWriter(new File(outputFile))

    pw.println(attrChangeRanges.mkString(","))

    pw.close()
  }

  def getAttrChangeThresholds(attrChangePercentage: Double): Array[Double] = {
    val line = Source.fromFile(outputFile).getLines.next()
    val attrChangeThresholds = line.split(",", -1).map(_.toDouble * (attrChangePercentage / (2 * 100))) // 2 for positive & negative change
    attrChangeThresholds
  }

}