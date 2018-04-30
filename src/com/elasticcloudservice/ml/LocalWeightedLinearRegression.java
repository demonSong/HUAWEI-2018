package com.elasticcloudservice.ml;

import com.elasticcloudservice.arima.Matrix;

public class LocalWeightedLinearRegression {
	
	double[][] X;
	double[] y;
	
	double sigma;
	
	public LocalWeightedLinearRegression(double[][] X, double[] y, double sigma){
		this.X = X;
		this.y = y;
		this.sigma = sigma;
	}
	
	public double[][] eye(int m){
		double[][] ret = new double[m][m];
		for (int i = 0; i < m; ++i) {
			ret[i][i] = 1;
		}
		return ret;
	}
	
	public double fit(double[] x){
		int m = X.length;
		int n = X[0].length;
		double[][] weight = eye(m);
		for (int i = 0; i < m; ++i) {
			double[] error = new double[n];
			for (int j = 0; j < n; ++j) error[j] = this.X[i][j] - x[j];
			double error_ = 0;
			for (int j = 0; j < n; ++j) error_ += error[j] * error[j];
			weight[i][i] = Math.exp(error_ / (-2.0 * this.sigma * this.sigma));
		}
		double[][] label = new double[y.length][1];
		for (int i = 0; i < y.length; ++i) label[i][0] = y[i];
		
		Matrix X_matrix = new Matrix(this.X);
		Matrix y_matrix = new Matrix(label);
		Matrix w_matrix = new Matrix(weight);
		
		Matrix xTWx = X_matrix.transpose().times(w_matrix.times(X_matrix));
		if (Double.compare(xTWx.det(), 0.0) == 0) {
			return -0x3f3f3f3f;
		}
		double[][] ws;
		try {
			ws = xTWx.inverse().times(X_matrix.transpose().times(w_matrix.times(y_matrix))).getArray();
		} catch (Exception e) {
			return -0x3f3f3f3f;
		}
		
		double pred = 0.0;
		for (int i = 0; i < n; ++i) {
			pred += x[i] * ws[i][0];
		}
		
		return pred;
	}
	
	public static void main(String[] args) {
		double[][] X = {{1},{2},{3},{4},{5},{6},{7},{8}, {9}, {10}, {11}};
		double[] y = {2, 4, 6, 8, 10, 12, 14, 16, 1.1, 0, 0};
		LocalWeightedLinearRegression lwlr = new LocalWeightedLinearRegression(X, y, 0.0); // k越大 y 越小
		lwlr.fit(new double[] {1.0});
		System.out.println(0.0);
	}
}
