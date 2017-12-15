package com.icanindya.adversarial.iscx

import com.icanindya.adversarial.Spark
import java.text.SimpleDateFormat
import java.util.Date
import java.util.TimeZone

object FeatureProcessor {

  val FLOW_FEATURES_FILE = "E:/Data/ISCX IDS/flowtbag/features_%d.ftb" //.format(int)
  val MODIFIED_FLOW_FEATURES_FILE = "E:/Data/ISCX IDS/flowtbag/features_%d/" //.format(int)

  def main(args: Array[String]): Unit = {

    val sc = Spark.getContext()

    for (i <- 11 to 17) {
      val lines = sc.textFile(FLOW_FEATURES_FILE.format(i))
      lines.map { line =>
        val tokens = line.split(",", -1)
        val featuresBuffer = tokens.toBuffer

        val startTimestamp = featuresBuffer.remove(44)
        val endTimestamp = featuresBuffer.remove(44)

        val startTime = timestampToLocaltime(startTimestamp.toLong, "GMT-3")
        val endTime = timestampToLocaltime(endTimestamp.toLong, "GMT-3")

        featuresBuffer.insert(5, startTime, endTime)

        featuresBuffer.mkString(",")
      }
      .coalesce(1).saveAsTextFile(MODIFIED_FLOW_FEATURES_FILE.format(i))
    }
  }

  def timestampToLocaltime(timeStamp: Long, timeZone: String): String = {
    val unixSeconds = timeStamp
    val date = new Date(unixSeconds * 1000L); // *1000 is to convert seconds to milliseconds
    val sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"); // the format of your date
    sdf.setTimeZone(TimeZone.getTimeZone(timeZone)); // give a timezone reference for formating (see comment at the bottom
    val formattedDate = sdf.format(date);
    formattedDate
  }

}