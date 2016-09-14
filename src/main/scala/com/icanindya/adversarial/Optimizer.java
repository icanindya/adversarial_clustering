package com.icanindya.adversarial;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

import ilog.concert.*;
import ilog.cplex.*;

import java.util.Random;


public class Optimizer {

	double distThres;
	double[] attrChangeThres;
	double[][] projCenters, transMat;
	Random rd = new Random();
	
	int closestClusterIndex = -1;
	double closestClusterSqDist = -1;
	int attempt = 0, success = 0;
	PrintWriter pw, pwTime;
	String optiLogFile = "E:/Data/KDD99/opti_log";
	String timeLogFile = "E:/Data/KDD99/time_log";

	public Optimizer(double[][] transMat, double[][] projCenters,
			double distThres, double[] attrChangeThres) {

		this.transMat = transMat;
		this.projCenters = projCenters;
		this.distThres = distThres;
		this.attrChangeThres = attrChangeThres;
		
		try {
			pw = new PrintWriter(new FileWriter(optiLogFile), true);
			pwTime = new PrintWriter(new FileWriter(timeLogFile), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public void setClosestClusterInfo(int index, double sqDist){
		this.closestClusterIndex = index;
		this.closestClusterSqDist = sqDist;
	}

	public boolean optimize(double[] origX, double[] projX) {
		
		try {
			pw = new PrintWriter(new FileWriter(optiLogFile, true), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		attempt++;

		// parameters
		// number of clusters
		int K = projCenters.length;

		// number of features
		int N = origX.length;

		// number of principal components
		int M = projX.length;

		int currCluster = -1;
		double currObjVal = Double.MAX_VALUE;
		double currOrigXp[] = new double[N];
		
		IloCplex cplex = null;
		for (int k = 0; k < K; k++) {
			try {
				// define new model
				
				cplex = new IloCplex();
				cplex.setOut(null);

				// variables

				// modified attack point
				IloNumVar[] origXp = cplex.numVarArray(N, 0, 1);
				
				// objective

				IloNumExpr objective = cplex.numExpr();
				IloNumExpr numExprs[] = new IloNumExpr[M];

				for (int j = 0; j < M; j++) {
					IloLinearNumExpr numExpr = cplex.linearNumExpr();
					for (int i = 0; i < N; i++) {
						numExpr.addTerm(origXp[i], transMat[i][j]);
					}
					numExprs[j] = cplex.square(cplex.diff(numExpr, projX[j]));
				}

				objective = cplex.sum(numExprs);
				cplex.addMinimize(objective);
				
				// constraints
				// constraint 1
				IloNumExpr numExprs2[] = new IloNumExpr[M];

				for (int j = 0; j < M; j++) {
					IloLinearNumExpr numExpr = cplex.linearNumExpr();
					for (int i = 0; i < N; i++) {
						numExpr.addTerm(origXp[i], transMat[i][j]);
					}
					numExprs2[j] = cplex.square(cplex.diff(numExpr, projCenters[k][j]));
				}
				
				cplex.addLe(cplex.sum(numExprs2), Math.pow(distThres, 2));
				
				// constraint 2
				for (int i = 0; i < N; i++) {
//					long start = System.currentTimeMillis();
					cplex.addLe(cplex.square(cplex.diff(origXp[i], origX[i])), Math.pow(attrChangeThres[i], 2));
//					long end = System.currentTimeMillis();
//					System.out.println(String.format("Time required: %d", end - start));
//					pwTime.println(String.format("%d", end - start));
				}

				if (cplex.solve()) {
					if (cplex.getObjValue() < currObjVal) {
						currCluster = k;
						currObjVal = cplex.getObjValue();
						currOrigXp = cplex.getValues(origXp);
					}
				}
				
			} catch (Exception e) {
				System.out.println(e.toString());
			}
			finally{
				if(cplex != null) cplex.end();
			}
		}
		System.out.println(String.format("Closest cluster %d, squared distance: %f", closestClusterIndex, closestClusterSqDist));
		pw.println(String.format("Closest cluster %d, squared distance: %f", closestClusterIndex, closestClusterSqDist));
		System.out.println(String.format("Attack point: %s", Arrays.toString(origX)));
		pw.println(String.format("Attack point: %s", Arrays.toString(origX)));
		System.out.println(String.format("Projected attack point: %s", Arrays.toString(projX)));
		pw.println(String.format("Projected attack point: %s", Arrays.toString(projX)));
		
		
		Boolean ret = false;
		
		if (currCluster != -1) {
			
			success++;
			
			System.out.println(String.format("Projected optimum cluster = %d: %s", currCluster, Arrays.toString(projCenters[currCluster])));
			pw.println(String.format("Projected optimum cluster = %d: %s", currCluster, Arrays.toString(projCenters[currCluster])));
			System.out.println(String.format("Modified attack point: %s", Arrays.toString(currOrigXp)));
			pw.println(String.format("Modified attack point: %s", Arrays.toString(currOrigXp)));

			ret = true;
			
		} else {
			
			System.out.println("Solution not found.");
			pw.println("Solution not found.");
			
		}
		
		System.out.println(String.format("Attempt: %d, Success: %d, Rate: %.2f%%", attempt, success, (success * 100.0)/attempt));
		pw.println(String.format("Attempt: %d, Success: %d, Rate: %.2f%%", attempt, success, (success * 100.0)/attempt));
		System.out.println();
		pw.println();
		
		pw.flush();
		pw.close();
		
		pwTime.flush();
		pwTime.close();
		
		return ret;
	}
}
