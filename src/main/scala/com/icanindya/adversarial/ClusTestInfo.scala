package com.icanindya.adversarial

class ClusTestInfo extends Serializable{
    var minSqDist = Double.MaxValue;
    var maxSqDist = Double.MinValue;
    var numAssocTestPoints = 0L;
    var totSqDist = 0.0;
  }