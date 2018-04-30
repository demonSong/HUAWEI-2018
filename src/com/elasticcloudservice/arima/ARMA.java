package com.elasticcloudservice.arima;

import java.util.*;

public class ARMA {

	double[] stdoriginalData = {};
	int p;
	int q;

	public ARMA(double[] stdoriginalData, int p, int q) {
		this.stdoriginalData = stdoriginalData;
		this.p = p;
		this.q = q;
	}

	public Vector<double[]> ARMAmodel() {
		Vector<double[]> vec = new Vector<>();
		double[] armaCoe = new ARMAMethod().computeARMACoe(this.stdoriginalData, this.p, this.q);
		double[] arCoe = new double[this.p + 1];
		System.arraycopy(armaCoe, 0, arCoe, 0, arCoe.length);
		double[] maCoe = new double[this.q + 1];
		System.arraycopy(armaCoe, (this.p + 1), maCoe, 0, maCoe.length);

		vec.add(arCoe);
		vec.add(maCoe);
		return vec;
	}
}
