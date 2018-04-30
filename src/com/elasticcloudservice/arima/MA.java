package com.elasticcloudservice.arima;

import java.util.Vector;

public class MA {

	double[] stdoriginalData = {};
	int q;
	ARMAMath armamath = new ARMAMath();

	public MA(double[] stdoriginalData, int q) {
		this.stdoriginalData = stdoriginalData;
		this.q = q;
	}

	public Vector<double[]> MAmodel() {
		Vector<double []>vec = new Vector<>();
		double [] maCoe = new ARMAMethod().computeMACoe(this.stdoriginalData, this.q);
		vec.add(maCoe);
		return vec;
	}
}
