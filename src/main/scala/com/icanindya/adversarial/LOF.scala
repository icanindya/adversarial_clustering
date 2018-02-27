package com.icanindya.adversarial
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import scala.util.Random

object LOF {
  
  var minPts = 5
  
  def setMinPts(k: Int){
    minPts = k
  }
  
  def main(args: Array[String]){
    println(randomizedSelect(Array(6, 7, 2, 1, 5, 7, 0, 10), 0, 7, 4))
  }
  
  
  def getLOF(targetPoint: Vector, points: RDD[Vector]){
   
    val otherPoints = points.filter(_ != targetPoint)
    
    val ptDists = otherPoints.map{ pt =>
      (pt, Vectors.sqdist(targetPoint, pt))
    }
    
    ptDists.cache()
    
  }
  
  
  def getLocalReachDensity(targetPoint: Vector, points: RDD[Vector]){
    
    val otherPoints = points.filter(_ != targetPoint)
    
    val ptDists = otherPoints.map{ pt =>
      (pt, Vectors.sqdist(targetPoint, pt))
    }
    
    ptDists.cache()
    
    val reachabilityDist = randomizedSelect(ptDists.map(_._2).collect, 0, ptDists.map(_._2).count.toInt - 1 , minPts)
    
    val neighborAndDists = ptDists.filter(_._2 <= reachabilityDist)
    
    val neighbors = neighborAndDists.map(_._1)
    val distances = neighborAndDists.map(_._2)
    
    neighbors.count/distances.sum
  }
  
  
  
  def randomizedSelect(distances: Array[Double], p: Int, r: Int, i: Int): Double = {
    if(p == r) return distances(p)
    val q = randomizedPartition(distances, p, r)
    val k = q - p + 1
    if(i == k) return distances(q)
    else if(i < k) return randomizedSelect(distances, p, q - 1, i)
    else return randomizedSelect(distances, q + 1, r, i - k)
  }
  
  def randomizedPartition(distances: Array[Double], p: Int, r: Int): Int = {
    val i = p + Random.nextInt(r - p + 1)
    exchange(distances, r, i)
    return partition(distances, p, r)
  }
  
  def partition(distances: Array[Double], p: Int, r: Int): Int = {
    val x = distances(r)
    var i = p - 1
    for (j <- p to r - 1){
      if(distances(j) <= x){
        i = i + 1
        exchange(distances, i, j)
      }
    }
    exchange(distances, i+1, r)
    return i + 1
  }
  
  def exchange[T](arr: Array[T], i: Int, j: Int){
    val temp = arr(i)
    arr(i) = arr(j)
    arr(j) = temp
  }
  
}