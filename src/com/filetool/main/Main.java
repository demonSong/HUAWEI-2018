package com.filetool.main;

import java.util.ArrayList;
import java.util.List;

import com.elasticcloudservice.predict.Predict;
import com.elasticcloudservice.train.Eval;
import com.filetool.util.FileUtil;
import com.filetool.util.LogUtil;

/**
 * 
 * 工具入口
 * 
 * @version [版本号, 2017-12-8]
 * @see [相关类/方法]
 * @since [产品/模块版本]
 */
public class Main {
	
	static List<int[]> params = new ArrayList<>();
	public static void dfs(int pos, int[] path) {
		if (pos == 15) {
			int[] aux = new int[path.length];
			System.arraycopy(path, 0, aux, 0, aux.length);
			params.add(aux);
			return;
		}
		else {
			for (int val = 4; val <= 6; ++val) {
				path[pos] = val;
				dfs(pos + 1, path);
			}
		}
	}
	
	public static void largerCase() {
		double sum = 0;
		int n = 1;
		int c = 0;
		for (int i = 0; i < n; ++i) {
			String ecsDataPath = "./newData/TrainData.txt";
			String inputFilePath = "./newData/input.txt";
			String resultFilePath = "./newData/output_2.txt";
			String testfile = "./newData/TestData.txt";
			
			LogUtil.printLog("Begin");
			Predict.seed = i + 2018;
			
			// 读取输入文件
			String[] ecsContent = FileUtil.read(ecsDataPath, null);
			String[] inputContent = FileUtil.read(inputFilePath, null);
	
			// 功能实现入口
			String[] resultContents = Predict.predictVm(ecsContent, inputContent);
	
			// 写入输出文件
			if (hasResults(resultContents)) {
				FileUtil.write(resultFilePath, resultContents, false);
			} else {
				FileUtil.write(resultFilePath, new String[] { "NA" }, false);
			}
			LogUtil.printLog("End");
			double s = Eval.score(testfile,inputFilePath,resultFilePath);
			if (!Double.isFinite(s)) continue;
			sum += s;
			c ++;
		}
		System.out.println("均值： " + sum / c);
	}
	
	public static void smallCase() {
		double sum = 0;
		int n = 100;
		int c = 0;
		for (int i = 0; i < n; ++i) {
//			if (i != 4) continue; 
			String ecsDataPath = "./newData/TrainData_2015.12.txt";
			String inputFilePath = "./newData/input_3hosttypes_5flavors_1week.txt";
			String resultFilePath = "./newData/output.txt";
			String testfile = "./newData/TestData_2016.1.8_2016.1.14.txt";
			
			LogUtil.printLog("Begin");
			Predict.seed = i + 2018;
			
			// 读取输入文件
			String[] ecsContent = FileUtil.read(ecsDataPath, null);
			String[] inputContent = FileUtil.read(inputFilePath, null);
	
			// 功能实现入口
			String[] resultContents = Predict.predictVm(ecsContent, inputContent);
	
			// 写入输出文件
			if (hasResults(resultContents)) {
				FileUtil.write(resultFilePath, resultContents, false);
			} else {
				FileUtil.write(resultFilePath, new String[] { "NA" }, false);
			}
			LogUtil.printLog("End");
			double s = Eval.score(testfile,inputFilePath,resultFilePath);
			if (!Double.isFinite(s)) continue;
			sum += s;
			c ++;
		}
		System.out.println("均值： " + sum / c);
	}
	
	@Deprecated
	public static void oneEval() {
		double sum = 0;
		int n = 100;
		for (int i = 0; i < n; ++i) {
			String ecsDataPath = "./data/Traindata.txt";
			String inputFilePath = "./data/input.txt";
			String resultFilePath = "./data/output.txt";
			String testfile = "./data/TestData.txt";
	
			LogUtil.printLog("Begin");
	
			// 读取输入文件
			String[] ecsContent = FileUtil.read(ecsDataPath, null);
			String[] inputContent = FileUtil.read(inputFilePath, null);
	
			// 功能实现入口
			String[] resultContents = Predict.predictVm(ecsContent, inputContent);
	
			// 写入输出文件
			if (hasResults(resultContents)) {
				FileUtil.write(resultFilePath, resultContents, false);
			} else {
				FileUtil.write(resultFilePath, new String[] { "NA" }, false);
			}
			LogUtil.printLog("End");
			double s = Eval.score(testfile,inputFilePath,resultFilePath);
			sum += s;
		}
		System.out.println("均值： " + sum / n);
	}
	
	@Deprecated
	public static void officalEval() {
		String root_path = "./eval/";
		String[] folders = {"2015_01/", "2015_02/", "2015_03/", "2015_04/", "2015_05/", "2015_12/", "2016_01/"};
		
		double sum = 0.0;
		for (String folder : folders) {
			String ecsDataPath = root_path + folder + "train.txt";
			String inputFilePath = root_path + folder + "input.txt";
			String resultFilePath = root_path + folder + "output.txt";
			String testFilePath = root_path + folder + "test.txt";
			
			LogUtil.printLog("Begin");
			
			// 读取输入文件
			String[] ecsContent = FileUtil.read(ecsDataPath, null);
			String[] inputContent = FileUtil.read(inputFilePath, null);
	
			// 功能实现入口
			String[] resultContents = Predict.predictVm(ecsContent, inputContent);
	
			// 写入输出文件
			if (hasResults(resultContents)) {
				FileUtil.write(resultFilePath, resultContents, false);
			} else {
				FileUtil.write(resultFilePath, new String[] { "NA" }, false);
			}
			LogUtil.printLog("End");
			
			double s = Eval.score(testFilePath,inputFilePath,resultFilePath);
			sum += s;
		}
		System.out.println("平均得分： " + sum / 7);
	}
	
	public static void main(String[] args) {
		long start = System.currentTimeMillis();
//		oneEval();
//		officalEval();
//		smallCase();
		largerCase();
		long end = System.currentTimeMillis();
		System.out.println("所花时间：" + (end - start) * 1.0 / 1000 + "s");
	}

	private static boolean hasResults(String[] resultContents) {
		if (resultContents == null) {
			return false;
		}
		for (String contents : resultContents) {
			if (contents != null && !contents.trim().isEmpty()) {
				return true;
			}
		}
		return false;
	}

}
