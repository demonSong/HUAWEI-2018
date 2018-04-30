package com.elasticcloudservice.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LSTM {
	
	int mem_cell_cnt;
	int x_dim;
	double lr;
	
	LSTMParam param;
	List<LSTMNode> lstm_node_list;
	List<double[]> x_list;
	
	public LSTM(int mem_cell_cnt, int x_dim, double lr) {
		this.mem_cell_cnt = mem_cell_cnt;
		this.x_dim = x_dim;
		this.lr = lr;
		
		param = new LSTMParam(mem_cell_cnt, x_dim);
		
		this.lstm_node_list = new ArrayList<>();
		this.x_list = new ArrayList<>();
	}
	
	public void clear() {
		x_list.clear();
	}
	
	
	public double y_list_is(double[] y, ToyLossLayer lossLayer) {
		assert y.length == x_list.size();
		int idx = this.x_list.size() - 1;
		
		double loss = lossLayer.loss(this.lstm_node_list.get(idx).state.h, y[idx]);
		double[] diff_h = lossLayer.bottom_diff(this.lstm_node_list.get(idx).state.h, y[idx]);
		double[] diff_s = new double[this.mem_cell_cnt];
		this.lstm_node_list.get(idx).top_diff_is(diff_h, diff_s);
		idx -= 1;
		
		while (idx >= 0) {
			loss += lossLayer.loss(this.lstm_node_list.get(idx).state.h, y[idx]);
			diff_h = lossLayer.bottom_diff(this.lstm_node_list.get(idx).state.h, y[idx]);
			diff_h = add(diff_h, this.lstm_node_list.get(idx + 1).state.bottom_diff_h);
			diff_s = this.lstm_node_list.get(idx + 1).state.bottom_diff_s;
			this.lstm_node_list.get(idx).top_diff_is(diff_h, diff_s);
			idx -= 1;
		}
		
		return loss;
	}
	
	public static double[] list2array(List<Float> list) {
		int n = list.size();
		double[] ret = new double[n];
		for (int i = 0; i < n; ++i) ret[i] = list.get(i);
		return ret;
	}
	
//	public void train(List<List<Double>> X, List<List<Double>> y, int epoch) {
//		for (int cur_iter = 0; cur_iter < epoch; ++cur_iter) {
//			double loss = 0;
//			for (int j = 0; j < X.size(); ++j) {
//				List<Double> x = X.get(j);
//				List<Double> y_list = y.get(j);
//				
//				// feed data
//				for (int i = 0; i < y_list.size(); ++i) {
//					this.x_list_add(new double[] {x.get(i)});
//				}
//				
//				loss += this.y_list_is(list2array(y_list), new ToyLossLayer());
//				this.clear();
//			}
//			this.param.apply_diff(lr);
//		}
//	}
	
//	public void train(List<List<Double>> X, List<List<Double>> y, int epoch) {
//		// 全局数据更新
//		for (int cur_iter = 0; cur_iter < epoch; ++cur_iter) {
//			for (int j = 0; j < X.size(); ++j) {
//				List<Double> x = X.get(j);
//				List<Double> y_list = y.get(j);
//				
//				// feed data
//				for (int i = 0; i < y_list.size(); ++i) {
//					this.x_list_add(new double[] {x.get(i)});
//				}
//				List<Double> pred = new ArrayList<>();
//				for (int i = 0; i < y_list.size(); ++i) {
//					pred.add(this.lstm_node_list.get(i).state.h[0]);
//				}
//				
//				double loss = this.y_list_is(list2array(y_list), new ToyLossLayer());
//				this.param.apply_diff(lr);
//				this.clear();
//			}
//		}
// 	}
	
//	public void train(List<List<Double>> X, List<List<Double>> y, int epoch) {
//		// 全局数据更新
//		for (int j = 0; j < X.size(); ++j) {
//			List<Double> x = X.get(j);
//			List<Double> y_list = y.get(j);
//			for (int cur_iter = 0; cur_iter < epoch; ++cur_iter) {
//				// feed data
//				for (int i = 0; i < y_list.size(); ++i) {
//					this.x_list_add(new double[] {x.get(i)});
//				}
//				List<Double> pred = new ArrayList<>();
//				for (int i = 0; i < y_list.size(); ++i) {
//					pred.add(this.lstm_node_list.get(i).state.h[0]);
//				}
//
//				double loss = this.y_list_is(list2array(y_list), new ToyLossLayer());
//				this.param.apply_diff(lr);
//				this.clear();
//			}
//		}
// 	}
	
	public void train(List<List<List<Float>>> X, List<List<Float>> y, int epoch) {
		// 全局数据更新
		for (int j = 0; j < X.size(); ++j) {
			List<List<Float>> x = X.get(j);
			List<Float> y_list = y.get(j);
			for (int cur_iter = 0; cur_iter < epoch; ++cur_iter) {
				// feed data
				for (int i = 0; i < y_list.size(); ++i) {
					this.x_list_add(list2array(x.get(i)));
				}
				List<Double> pred = new ArrayList<>();
				for (int i = 0; i < y_list.size(); ++i) {
					pred.add(this.lstm_node_list.get(i).state.h[0]);
				}

				double loss = this.y_list_is(list2array(y_list), new ToyLossLayer());
				this.param.apply_diff(lr);
				this.clear();
			}
		}
 	}
	
	public float predict(List<List<Float>> X) {
		for (int i = 0; i < X.size(); ++i) {
			this.x_list_add(list2array(X.get(i)));
		}
		double pred = this.lstm_node_list.get(X.size() - 1).state.h[0];
		this.clear();
		return (float) pred;
	}
	
//	public double predict(List<Double> X) {
//		for (int i = 0; i < X.size(); ++i) {
//			this.x_list_add(new double[] {X.get(i)});
//		}
//		double pred = this.lstm_node_list.get(X.size() - 1).state.h[0];
//		this.clear();
//		return pred;
//	}
	
	
	public void x_list_add(double[] x) { // 一个样例的数据？
		x_list.add(x);
		if (x_list.size() > lstm_node_list.size()) {
			LSTMState state = new LSTMState(this.mem_cell_cnt, this.x_dim);
			lstm_node_list.add(new LSTMNode(state, this.param));
		}
		
		int idx = x_list.size() - 1; 
		if (idx == 0) {
			this.lstm_node_list.get(idx).bottom_data_is(x, null, null);
		}
		else {
			double[] s_prev = this.lstm_node_list.get(idx - 1).state.s;
			double[] h_prev = this.lstm_node_list.get(idx - 1).state.h;
			this.lstm_node_list.get(idx).bottom_data_is(x, s_prev, h_prev);
		}
	}
	
	public static double[] hstack(double[] a, double[] b) {
		double[] ret = new double[a.length + b.length];
		int k = 0;
		for (int i = 0; i < a.length; ++i) ret[k++] = a[i];
		for (int i = 0; i < b.length; ++i) ret[k++] = b[i];
		return ret;
	}
	
	public static double[][] hstack(double[][] a, double[][] b) {
		double[][] ret = new double[a.length][];
		for (int i = 0; i < a.length; ++i) ret[i] = hstack(a[i], b[i]);
		return ret;
	}
	
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	public static double[] sigmoid(double[] a) {
		double[] b = new double[a.length];
		for (int i = 0; i < a.length; ++i) b[i] = sigmoid(a[i]);
		return b;
	}
	
	public static double sigmoid_derivative(double v) {
		return v * (1 - v);
	}
	
	public static double[] sigmoid_derivative(double[] v) {
		double[] ret = new double[v.length];
		for (int i = 0; i < v.length; ++i) ret[i] = sigmoid_derivative(v[i]);
		return ret;
	}
	
	public static double tanh_derivative(double v) {
		return 1 - v * v;
	}
	
	public static double[] tanh_derivative(double[] v) {
		double[] ret = new double[v.length];
		for (int i = 0; i < v.length; ++i) ret[i] = tanh_derivative(v[i]);
		return ret;
	}
	
	public static double[][] rand_arr(double a, double b, int x, int y){
		double[][] ret = new double[x][y];
		Random random = new Random(2016666);
		for (int i = 0; i < x; ++i) {
			for (int j = 0; j < y; ++j) {
				ret[i][j] = random.nextDouble() * (b - a) + a;
			}
		}
		return ret;
	}
	
	public static double[] rand_vec(double a, double b, int x){
		double[] ret = new double[x];
		Random random = new Random(2016666);
		for (int i = 0; i < x; ++i) {
			ret[i] = random.nextDouble() * (b - a) + a;
		}
		return ret;
	}
	
	public static double[] zero_like(double[] a) {
		double[] b = new double[a.length];
		return b;
	}
	
	public static double[][] zero_like(double[][] a) {
		double[][] b = new double[a.length][a[0].length];
		return b;
	}
	
	public static double dot(double[] a, double[] b) {
		double sum = 0.0;
		for (int i = 0; i < a.length; ++i) {
			sum += a[i] * b[i];
		}
		return sum;
	}
	
	public static double[] dot(double[][] a, double[] b) {
		double[] ret = new double[a.length];
		for (int i = 0; i < a.length; ++i) {
			ret[i] = dot(a[i], b);
		}
		return ret;
	}
	
	public static double[] mat(double[] a, double[] b) {
		double[] ret = new double[a.length];
		for (int i = 0; i < a.length; ++i) ret[i] = a[i] * b[i];
		return ret;
	}
	
	public static double[][] transpose(double[][] a){
		int n = a.length;
		int m = a[0].length;
		double[][] ret = new double[m][n];
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				ret[i][j] = a[j][i];
			}
		}
		return ret;
	}
	
	public static double[] add(double[] a, double[] b) {
		double[] ret = new double[a.length];
		for (int i = 0; i < a.length; ++i) ret[i] = a[i] + b[i];
		return ret;
	}
	
	public static double[][] add(double[][] a, double[][] b){
		double[][] ret = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; ++i) {
			for (int j = 0; j < a[0].length; ++j) {
				ret[i][j] = a[i][j] + b[i][j];
			}
		}
		return ret;
	}
	
	/**
	 * 
	 * @param a [1, 2, 3]
	 * @param b [1, 1, 1, 1]
	 * @return
	 * [[1, 1, 1, 1]
	 * ,[2, 2, 2, 2]
	 * ,[3, 3, 3, 3]]
	 * 
	 */
	public static double[][] outer(double[] a, double[] b){
		int n = a.length;
		int m = b.length;
		double[][] ret = new double[n][m];
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				ret[i][j] = a[i] * b[j];
			}
		}
		return ret;
	}
	
	/**
	 * a[l, r)
	 * @param a
	 * @param l
	 * @param r
	 * @return
	 */
	public static double[] dim(double[] a, int l, int r) {
		int len = r - l;
		double[] ret = new double[len];
		for (int i = l; i < r; ++i) {
			ret[i - l] = a[i];
		}
		return ret;
	}
			
	public static double[] WtxPlusBias(double[][] w, double[] x, double[] b) {
		int n = w.length;
		double[] ans = new double[n];
		for (int i = 0; i < n; ++i) {
			double wtx = dot(w[i], x);
			ans[i] = wtx + b[i];
		}
		return ans;
	}
	
	public static double[] tanh(double[] a) {
		double[] b = new double[a.length];
		for (int i = 0; i < a.length; ++i) b[i] = Math.tanh(a[i]);
		return b;
	}
	
	public static void main(String[] args) {
		Random random = new Random(2016666);
		int mem_cell_cnt = 100;
		int x_dim = 50;
		LSTM lstm = new LSTM(mem_cell_cnt, x_dim, 0.1);
		
		// input
		double[] y = {-0.5, 0.6, 0.3, -0.5};
		double[][] X = new double[y.length][x_dim]; // x_1, x_2, x_3, x_4, ... , x_50
		
		for (int i = 0; i < X.length; ++i) {
			for (int j = 0; j < X[0].length; ++j) {
				X[i][j] = random.nextDouble();
			}
		}
		
		for (int cut_iter = 0; cut_iter < 1000; ++cut_iter) {
			System.out.print("iter: " + cut_iter + ": ");
			for (int i = 0; i < y.length; ++i) {
				lstm.x_list_add(X[i]);
			}
			String[] predict = new String[y.length];
			for (int i = 0; i < y.length; ++i) {
				predict[i] = lstm.lstm_node_list.get(i).state.h[0] + "";
			}
			System.out.print("y_pred = [" + String.join(",", predict) + "], ");
			double loss = lstm.y_list_is(y, new ToyLossLayer());
			lstm.param.apply_diff(0.1);
			System.out.println("loss: " + loss);
			lstm.clear();
		}
		
	}
}

class ToyLossLayer{
	
	/**
	 * compute square loss with first element of hidden layer array
	 */
	
	public double loss(double[] pred, double label) { // 第一个记忆单元的输出？
		return (pred[0] - label) * (pred[0] - label);
	}
	
	public double[] bottom_diff(double[] pred, double label) {
		double[] diff = new double[pred.length];
		diff[0] = 2 * (pred[0] - label);
		return diff;
	}
	
}

class LSTMParam{
	
	int mem_cell_cnt;
	int x_dim;
	int concat_len;
	
	double[][] wg, wi, wf, wo;
	double[][] wg_diff, wi_diff, wf_diff, wo_diff;
	double[] bg, bi, bf, bo;
	double[] bg_diff, bi_diff, bf_diff, bo_diff;
	
	public LSTMParam(int mem_cell_cnt, int x_dim) {
		this.mem_cell_cnt = mem_cell_cnt;
		this.x_dim = x_dim;
		this.concat_len = mem_cell_cnt + x_dim;
		
		this.wg = LSTM.rand_arr(-0.1, 0.1, mem_cell_cnt, concat_len);
		this.wf = LSTM.rand_arr(-0.1, 0.1, mem_cell_cnt, concat_len);
		this.wi = LSTM.rand_arr(-0.1, 0.1, mem_cell_cnt, concat_len);
		this.wo = LSTM.rand_arr(-0.1, 0.1, mem_cell_cnt, concat_len);
		
		this.bg = LSTM.rand_vec(-0.1, 0.1, mem_cell_cnt);
		this.bf = LSTM.rand_vec(-0.1, 0.1, mem_cell_cnt);
		this.bo = LSTM.rand_vec(-0.1, 0.1, mem_cell_cnt);
		this.bi = LSTM.rand_vec(-0.1, 0.1, mem_cell_cnt);
		
		this.wg_diff = new double[mem_cell_cnt][concat_len];
		this.wo_diff = new double[mem_cell_cnt][concat_len];
		this.wi_diff = new double[mem_cell_cnt][concat_len];
		this.wf_diff = new double[mem_cell_cnt][concat_len];
		
		this.bg_diff = new double[mem_cell_cnt];
		this.bo_diff = new double[mem_cell_cnt];
		this.bi_diff = new double[mem_cell_cnt];
		this.bf_diff = new double[mem_cell_cnt];
	}
	
	public void apply_diff(double lr) {
		reduce(wg, wg_diff, lr);
		reduce(wf, wf_diff, lr);
		reduce(wo, wo_diff, lr);
		reduce(wi, wi_diff, lr);
		
		reduce(bf, bf_diff, lr);
		reduce(bg, bg_diff, lr);
		reduce(bo, bo_diff, lr);
		reduce(bi, bi_diff, lr);
		
		this.wg_diff = new double[mem_cell_cnt][concat_len];
		this.wo_diff = new double[mem_cell_cnt][concat_len];
		this.wi_diff = new double[mem_cell_cnt][concat_len];
		this.wf_diff = new double[mem_cell_cnt][concat_len];
		
		this.bg_diff = new double[mem_cell_cnt];
		this.bo_diff = new double[mem_cell_cnt];
		this.bi_diff = new double[mem_cell_cnt];
		this.bf_diff = new double[mem_cell_cnt];
	}
	
	private void reduce(double[][] w, double[][] w_diff, double lr) {
		int n = w.length;
		int m = w[0].length;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				w[i][j] -= lr * w_diff[i][j];
			}
		}
	}
	
	private void reduce(double[] b, double[] b_diff, double lr) {
		int n = b.length;
		for (int i = 0; i < n; ++i) {
			b[i] -= lr * b_diff[i];
		}
	}
}

class LSTMState{
	
	double[] g, i, f, o, s, h, bottom_diff_h, bottom_diff_s;
	
	public LSTMState(int mem_cell_cnt, int x_dim) {
		this.g = new double[mem_cell_cnt];
		this.i = new double[mem_cell_cnt];
		this.f = new double[mem_cell_cnt];
		this.o = new double[mem_cell_cnt];
		this.s = new double[mem_cell_cnt];
		this.h = new double[mem_cell_cnt];
		this.bottom_diff_h = new double[mem_cell_cnt];
		this.bottom_diff_s = new double[mem_cell_cnt];
	}
}

class LSTMNode{
	
	LSTMState state;
	LSTMParam param;
	
	double[] s_prev;
	double[] h_prev;
	
	double[] xc;
	
	public LSTMNode(LSTMState state, LSTMParam param) {
		this.state = state;
		this.param = param;
	}
	
	public void bottom_data_is(double[] x, double[] s_prev, double[] h_prev) {
		if (s_prev == null) s_prev = LSTM.zero_like(this.state.s);
		if (h_prev == null) h_prev = LSTM.zero_like(this.state.h);
		
		this.s_prev = s_prev;
		this.h_prev = h_prev;
		
		// concatenate x(t) and h(t - 1)
		this.xc = LSTM.hstack(x, h_prev);
		this.state.g = LSTM.tanh(LSTM.WtxPlusBias(this.param.wg, xc, this.param.bg));
		this.state.i = LSTM.sigmoid(LSTM.WtxPlusBias(this.param.wi, xc, this.param.bi));
		this.state.f = LSTM.sigmoid(LSTM.WtxPlusBias(this.param.wf, xc, this.param.bf));
		this.state.o = LSTM.sigmoid(LSTM.WtxPlusBias(this.param.wo, xc, this.param.bo));
		
		this.state.s = LSTM.add(LSTM.mat(this.state.g, this.state.i), LSTM.mat(s_prev, this.state.f));
		this.state.h = LSTM.mat(this.state.s, this.state.o);
		
	}
	
	public void top_diff_is(double[] top_diff_h, double[] top_diff_s) {
		double[] ds  = LSTM.add(top_diff_s, LSTM.mat(this.state.o, top_diff_h));
		double[] dot = LSTM.mat(this.state.s, top_diff_h);
		double[] di  = LSTM.mat(this.state.g, ds);
		double[] dg  = LSTM.mat(this.state.i, ds);
		double[] df  = LSTM.mat(this.s_prev, ds);
		
		double[] di_input = LSTM.mat(LSTM.sigmoid_derivative(this.state.i), di);
		double[] df_input = LSTM.mat(LSTM.sigmoid_derivative(this.state.f), df);
		double[] do_input = LSTM.mat(LSTM.sigmoid_derivative(this.state.o), dot);
		double[] dg_input = LSTM.mat(LSTM.tanh_derivative(this.state.g), dg);
		
		this.param.wi_diff = LSTM.add(this.param.wi_diff, LSTM.outer(di_input, this.xc));
		this.param.wf_diff = LSTM.add(this.param.wf_diff, LSTM.outer(df_input, this.xc));
		this.param.wo_diff = LSTM.add(this.param.wo_diff, LSTM.outer(do_input, this.xc));
		this.param.wg_diff = LSTM.add(this.param.wg_diff, LSTM.outer(dg_input, this.xc));
		
		this.param.bi_diff = LSTM.add(this.param.bi_diff, di_input);
		this.param.bf_diff = LSTM.add(this.param.bf_diff, df_input);
		this.param.bo_diff = LSTM.add(this.param.bo_diff, do_input);
		this.param.bg_diff = LSTM.add(this.param.bg_diff, dg_input);
		
		double[] dxc = LSTM.zero_like(this.xc);
		dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wi), di_input));
		dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wf), df_input));
		dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wo), do_input));
		dxc = LSTM.add(dxc, LSTM.dot(LSTM.transpose(this.param.wg), dg_input));
		
		this.state.bottom_diff_s = LSTM.mat(ds, this.state.f);
		this.state.bottom_diff_h = LSTM.dim(dxc, this.param.x_dim, dxc.length);
	}
}

