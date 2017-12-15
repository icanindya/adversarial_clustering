package com.icanindya.adversarial.iscx

import java.text.SimpleDateFormat
import java.util.TimeZone
import java.util.Date
import scala.io.Source
import java.io.PrintWriter

object test {
  def main(args: Array[String]): Unit = {
//    val unixSeconds = 1493335443;
//    val date = new Date(unixSeconds * 1000L); // *1000 is to convert seconds to milliseconds
//    val sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss"); // the format of your date
//    sdf.setTimeZone(TimeZone.getTimeZone("GMT-3")); // give a timezone reference for formating (see comment at the bottom
//    val formattedDate = sdf.format(date);
//    System.out.println(formattedDate);
    
    for(i <- 12 to 17){
      val inFile = "E:/Data/ISCX IDS/processed_labels/labels_%d.txt".format(i)
      val outFile = "E:/Data/ISCX IDS/processed_labels/label_%d.txt".format(i)
      val pw = new PrintWriter(outFile)
      
      try{
        for(line <- Source.fromFile(inFile).getLines()){
          val tokens = line.split(",", -1)
          tokens(5) = tokens(5).split("T").mkString(" ")
          tokens(6) = tokens(6).split("T").mkString(" ")
          pw.println(tokens.mkString(","))
        }
      }catch{
        case ex: Exception => println("An exception occured")
      }
      finally {
        pw.close()
      }
    }
    
    
    
  }
}