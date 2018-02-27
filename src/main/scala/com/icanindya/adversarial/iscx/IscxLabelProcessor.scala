package com.icanindya.adversarial.iscx

import org.apache.spark.sql.SQLContext
import com.icanindya.adversarial.Spark

object IscxLabelProcessor {

  val IN_LABEL_FILE_PATH = "E:/Data/ISCX_IDS/labeled_flows_xml/TestbedThuJun17-3Flows.xml"
  val ROW_TAG = "TestbedThuJun17-3Flows"
  val OUT_LABEL_FILE_PATH = "E:/Data/ISCX_IDS/labels"
  
  val protocolToNum = Map("tcp_ip" -> "6", "udp_ip" -> "17", "icmp_ip" -> "1", "igmp" -> "2")

  def main(args: Array[String]): Unit = {

    val sc = Spark.getContext()
    val sqlContext = new SQLContext(sc)
    val df = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", ROW_TAG).load(IN_LABEL_FILE_PATH)

    val records = df.select("source", "sourcePort", "destination", "destinationPort", "protocolName", "startDateTime", "stopDateTime", "Tag",
      "appName", "direction", "totalSourcePackets", "totalSourceBytes", "totalDestinationPackets",
      "totalDestinationBytes", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription" //    ,"sourcePayloadAsBase64","destinationPayloadAsBase64","sourcePayloadAsUTF","destinationPayloadAsUTF"
      )
      .rdd
      .map { r =>
         val protocolName = r.getAs[String]("protocolName")
         val protocolNum = if(protocolToNum.contains(protocolName)) protocolToNum(protocolName) else "" 
        
         var sourceFlags = r.getAs[String]("sourceTCPFlagsDescription")
         sourceFlags = if (sourceFlags != null) sourceFlags.replace(",", "-") else ""

         var destFlags = r.getAs[String]("destinationTCPFlagsDescription")
         destFlags = if (destFlags != null) destFlags.replace(",", "-") else ""
           
         val startDateTime = r.getAs[String]("startDateTime").split("T").mkString(" ")  
         val stopDateTime =  r.getAs[String]("stopDateTime").split("T").mkString(" ")
        val tuple = (r.getAs[String]("source"), r.getAs[String]("sourcePort"),
          r.getAs[String]("destination"), r.getAs[String]("destinationPort"),
          protocolNum, startDateTime, stopDateTime, r.getAs[String]("Tag"),
          r.getAs[String]("appName"), r.getAs[String]("direction"), 
          r.getAs[String]("totalSourcePackets"), r.getAs[String]("totalSourceBytes"),
          r.getAs[String]("totalDestinationPackets"), r.getAs[String]("totalDestinationBytes"), sourceFlags, destFlags)
          
        val accept = if(protocolNum != "") true else false
        
        (tuple, accept)
      }
      .filter(_._2 == true)
      .map(_._1.productIterator.mkString(","))
      
      records.coalesce(1).saveAsTextFile(OUT_LABEL_FILE_PATH)

  }
  
}