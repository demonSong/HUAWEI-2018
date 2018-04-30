package com.elasticcloudservice.predict;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.date.utils.DateUtil;
import com.elasticcloudservice.train.Allocation;
import com.elasticcloudservice.train.Eval;
import com.elasticcloudservice.train.FeatureExtract;
import com.elasticcloudservice.train.Train;

public class Predict {
	
	static final boolean DEBUG = true;
//	public static final int[] flavor_param = {
//			5, // flavor 1
//			5, // flavor 2
//			5, // flavor 3   0
//			5, // flavor 4   0
//			4, // flavor 5
//			5, // flavor 6   0
//			5, // flavor 7   0
//			5, // flavor 8
//			6, // flavor 9
//			5, // flavor 10  0
//			4, // flavor 11
//			5, // flavor 12  0
//			5, // flavor 13  0
//			5, // flavor 14  0
//			5, // flavor 15  0
//			5, // flavor 15  0
//			5, // flavor 16  
//			5, // flavor 17  
//			5, // flavor 18  
//	};

//	public static final int[] flavor_param = {
//			15, // flavor 1
//			15, // flavor 2
//			15, // flavor 3   0
//			15, // flavor 4   0
//			15, // flavor 5
//			15, // flavor 6   0
//			15, // flavor 7   0
//			15, // flavor 8
//			15, // flavor 9
//			15, // flavor 10  0
//			15, // flavor 11
//			15, // flavor 12  0
//			15, // flavor 13  0
//			15, // flavor 14  0
//			15, // flavor 15  0
//			15, // flavor 15  0
//			15, // flavor 16  
//			15, // flavor 17  
//			15, // flavor 18  
//	};
	
	public static int[] flavor_param(int param) {
		int[] flavor = new int[18];
		for (int i = 0; i < flavor.length; ++i) flavor[i] = param;
		return flavor;
	}
	
	
	public static int[] flavor_param_two = {
			5, // flavor 1
			5, // flavor 2
			5, // flavor 3   0
			5, // flavor 4   0
			5, // flavor 5
			5, // flavor 6   0
			5, // flavor 7   0
			5, // flavor 8
			5, // flavor 9
			5, // flavor 10  0
			5, // flavor 11
			5, // flavor 12  0
			5, // flavor 13  0
			5, // flavor 14  0
			5, // flavor 15  0
			5, // flavor 15  0
			5, // flavor 16  
			5, // flavor 17  
			5, // flavor 18  
	};
	
	
	public static final int[] linear_diff = {
			6, // flavor 1
			6, // flavor 2
			6, // flavor 3   0
			6, // flavor 4   0
			7, // flavor 5
			6, // flavor 6   0
			6, // flavor 7   0
			6, // flavor 8
			5, // flavor 9
			6, // flavor 10  0
			5, // flavor 11
			6, // flavor 12  0
			6, // flavor 13  0
			6, // flavor 14  0
			6, // flavor 15  0
			5, // flavor 16  
			5, // flavor 17  
			5, // flavor 18  
	};
	
	public static final int[] roll_param = {
			1, // flavor 1
			1, // flavor 2
			1, // flavor 3   0
			1, // flavor 4   0
			1, // flavor 5
			1, // flavor 6   0
			1, // flavor 7   0
			1, // flavor 8
			1, // flavor 9
			1, // flavor 10  0
			1, // flavor 11
			1, // flavor 12  0
			1, // flavor 13  0
			1, // flavor 14  0
			1, // flavor 15  0
			5, // flavor 16  
			5, // flavor 17  
			5, // flavor 18  
	};
	
	
	static final String[] spring = { "2012-01-22", "2012-01-23", "2012-01-24", "2012-01-25", "2012-01-26", "2012-01-27",
			"2012-01-28", "2013-02-09", "2013-02-10", "2013-02-11", "2013-02-12", "2013-02-13", "2013-02-14",
			"2013-02-15", "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03", "2014-02-04", "2014-02-05",
			"2014-02-06", "2015-02-18", "2015-02-19", "2015-02-20", "2015-02-21", "2015-02-22", "2015-02-23",
			"2015-02-24", "2016-02-07", "2016-02-08", "2016-02-09", "2016-02-10", "2016-02-11", "2016-02-12",
			"2016-02-13", "2017-01-27", "2017-01-28", "2017-01-29", "2017-01-30", "2017-01-31", "2017-02-01",
			"2017-02-02", "2018-02-15", "2018-02-16", "2018-02-17", "2018-02-18", "2018-02-19", "2018-02-20",
			"2018-02-21" };
	
	static final String[] national = { "2012-09-30", "2012-10-01", "2012-10-02", "2012-10-03", "2012-10-04",
			"2012-10-05", "2012-10-06", "2012-10-07", "2013-10-01", "2013-10-02", "2013-10-03", "2013-10-04",
			"2013-10-05", "2013-10-06", "2013-10-07", "2014-10-01", "2014-10-02", "2014-10-03", "2014-10-04",
			"2014-10-05", "2014-10-06", "2014-10-07", "2015-10-01", "2015-10-02", "2015-10-03", "2015-10-04",
			"2015-10-05", "2015-10-06", "2015-10-07", "2016-10-01", "2016-10-02", "2016-10-03", "2016-10-04",
			"2016-10-05", "2016-10-06", "2016-10-07", "2017-10-01", "2017-10-02", "2017-10-03", "2017-10-04",
			"2017-10-05", "2017-10-06", "2017-10-07", "2017-10-08", "2018-10-01", "2018-10-02", "2018-10-03",
			"2018-10-04", "2018-10-05", "2018-10-06", "2018-10-07" };
	
	static final String[] holidays_end = { "2012-01-03", "2012-01-28", "2012-04-04", "2012-05-01", "2012-06-24",
			"2012-10-07", "2013-01-03", "2013-02-15", "2013-04-06", "2013-05-01", "2013-06-12", "2013-09-21",
			"2013-10-07", "2014-01-01", "2014-02-06", "2014-04-07", "2014-05-03", "2014-06-02", "2014-09-08",
			"2014-10-07", "2015-01-03", "2015-02-24", "2015-04-06", "2015-05-03", "2015-06-22", "2015-09-05",
			"2015-09-27", "2015-10-07", "2016-01-03", "2016-02-13", "2016-04-04", "2016-05-02", "2016-06-11",
			"2016-09-17", "2016-10-07", "2017-01-02", "2017-02-02", "2017-04-04", "2017-05-01", "2017-10-08",
			"2018-01-01", "2018-02-21", "2018-04-07", "2018-05-01", "2018-06-18", "2018-09-24", "2018-10-07" };
	
	static final String[] holidays_start = { "2012-01-01", "2012-01-22", "2012-04-02", "2012-04-29", "2012-06-22",
			"2012-09-30", "2013-01-01", "2013-02-09", "2013-04-04", "2013-04-29", "2013-06-10", "2013-09-19",
			"2013-10-01", "2014-01-01", "2014-01-31", "2014-04-05", "2014-05-01", "2014-05-31", "2014-09-06",
			"2014-10-01", "2015-01-01", "2015-02-18", "2015-04-04", "2015-05-01", "2015-06-20", "2015-09-03",
			"2015-09-26", "2015-10-01", "2016-01-01", "2016-02-07", "2016-04-02", "2016-04-30", "2016-06-09",
			"2016-09-15", "2016-10-01", "2016-12-31", "2017-01-27", "2017-04-02", "2017-04-29", "2017-10-01",
			"2017-12-30", "2018-02-15", "2018-04-05", "2018-04-29", "2018-06-16", "2018-09-22", "2018-10-01" };
	
	static final String[] holidays = { "2012-01-01", "2012-01-02", "2012-01-03", "2012-01-22", "2012-01-23",
			"2012-01-24", "2012-01-25", "2012-01-26", "2012-01-27", "2012-01-28", "2012-04-02", "2012-04-03",
			"2012-04-04", "2012-04-29", "2012-04-30", "2012-05-01", "2012-06-22", "2012-06-23", "2012-06-24",
			"2012-09-30", "2012-10-01", "2012-10-02", "2012-10-03", "2012-10-04", "2012-10-05", "2012-10-06",
			"2012-10-07", "2013-01-01", "2013-01-02", "2013-01-03", "2013-02-09", "2013-02-10", "2013-02-11",
			"2013-02-12", "2013-02-13", "2013-02-14", "2013-02-15", "2013-04-04", "2013-04-05", "2013-04-06",
			"2013-04-29", "2013-04-30", "2013-05-01", "2013-06-10", "2013-06-11", "2013-06-12", "2013-09-19",
			"2013-09-20", "2013-09-21", "2013-10-01", "2013-10-02", "2013-10-03", "2013-10-04", "2013-10-05",
			"2013-10-06", "2013-10-07", "2014-01-01", "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03",
			"2014-02-04", "2014-02-05", "2014-02-06", "2014-04-05", "2014-04-06", "2014-04-07", "2014-05-01",
			"2014-05-02", "2014-05-03", "2014-05-31", "2014-06-01", "2014-06-02", "2014-09-06", "2014-09-07",
			"2014-09-08", "2014-10-01", "2014-10-02", "2014-10-03", "2014-10-04", "2014-10-05", "2014-10-06",
			"2014-10-07", "2015-01-01", "2015-01-02", "2015-01-03", "2015-02-18", "2015-02-19", "2015-02-20",
			"2015-02-21", "2015-02-22", "2015-02-23", "2015-02-24", "2015-04-04", "2015-04-05", "2015-04-06",
			"2015-05-01", "2015-05-02", "2015-05-03", "2015-06-20", "2015-06-21", "2015-06-22", "2015-09-03",
			"2015-09-04", "2015-09-05", "2015-09-26", "2015-09-27", "2015-10-01", "2015-10-02", "2015-10-03",
			"2015-10-04", "2015-10-05", "2015-10-06", "2015-10-07", "2016-01-01", "2016-01-02", "2016-01-03",
			"2016-02-07", "2016-02-08", "2016-02-09", "2016-02-10", "2016-02-11", "2016-02-12", "2016-02-13",
			"2016-04-02", "2016-04-03", "2016-04-04", "2016-04-30", "2016-05-01", "2016-05-02", "2016-06-09",
			"2016-06-10", "2016-06-11", "2016-09-15", "2016-09-16", "2016-09-17", "2016-10-01", "2016-10-02",
			"2016-10-03", "2016-10-04", "2016-10-05", "2016-10-06", "2016-10-07", "2016-12-31", "2017-01-01",
			"2017-01-02", "2017-01-27", "2017-01-28", "2017-01-29", "2017-01-30", "2017-01-31", "2017-02-01",
			"2017-02-02", "2017-04-02", "2017-04-03", "2017-04-04", "2017-04-29", "2017-04-30", "2017-05-01",
			"2017-05-28", "2017-05-29", "2017-05-30", "2017-10-01", "2017-10-02", "2017-10-03", "2017-10-04",
			"2017-10-05", "2017-10-06", "2017-10-07", "2017-10-08", "2017-12-30", "2017-12-31", "2018-01-01",
			"2018-02-15", "2018-02-16", "2018-02-17", "2018-02-18", "2018-02-19", "2018-02-20", "2018-02-21",
			"2018-04-05", "2018-04-06", "2018-04-07", "2018-04-29", "2018-04-30", "2018-05-01", "2018-06-16",
			"2018-06-17", "2018-06-18", "2018-09-22", "2018-09-23", "2018-09-24", "2018-10-01", "2018-10-02",
			"2018-10-03", "2018-10-04", "2018-10-05", "2018-10-06", "2018-10-07" };
	
	public static Set<String> holidaySet(){
		Set<String> set = new HashSet<>();
		for (String date : holidays) {
			set.add(date);
		}
		return set;
	}
	
	public static Set<Long> longHolidaySet(String[] holidays){
		Set<Long> set = new HashSet<>();
		for (String date : holidays) {
			set.add(DateUtil.year2dayLong(date));
		}
		return set;
	}
	
	public static Set<Long> longHolidaySet(){
		Set<Long> set = new HashSet<>();
		for (String date : holidays) {
			set.add(DateUtil.year2dayLong(date));
		}
		return set;
	}
	
	public static Map<String, Integer> chooseModel(List<Feature> features, List<Record> dataset, Param param, int bestModel){
		if (bestModel == Eval.MEAN_MODEL_4) {
			return Train.recentMean(features, param, 4, true);
		}
		if (bestModel == Eval.MEAN_MODEL_5) {
			return Train.recentMean(features, param, 5, true);
		}
		if (bestModel == Eval.MEAN_MODEL_6) {
			return Train.recentMean(features, param, 6, true);
		}
		if (bestModel == Eval.ARIMA_MEAN_AVG_4) {
			Map<String, Integer> model = new HashMap<>();
			Map<String, Integer> model_b = Train.arima(features, dataset, param, flavor_param(5), 1);
			Map<String, Integer> model_c = Train.recentMean(features, param, 4, true);
			
			for (String key : param.virtuals.keySet()) {
				double arima  = model_b.containsKey(key) ? model_b.get(key) : 0;
				double mean   = model_c.containsKey(key) ? model_c.get(key) : 0;
				
				int sum = (int) Math.ceil(( arima * 1 / 2 + mean * 1 / 2));
				model.put(key, sum >= 0 ? sum : 0);
			}
			return model;
		}
		if (bestModel == Eval.ARIMA_MEAN_AVG_5) {
			Map<String, Integer> model = new HashMap<>();
			Map<String, Integer> model_b = Train.arima(features, dataset, param, flavor_param(5), 1);
			Map<String, Integer> model_c = Train.recentMean(features, param, 5, true);
			
			for (String key : param.virtuals.keySet()) {
				double arima  = model_b.containsKey(key) ? model_b.get(key) : 0;
				double mean   = model_c.containsKey(key) ? model_c.get(key) : 0;
				
				int sum = (int) Math.ceil(( arima * 1 / 2 + mean * 1 / 2));
				model.put(key, sum >= 0 ? sum : 0);
			}
			return model;
		}
		if (bestModel == Eval.ARIMA_MEAN_AVG_6) {
			Map<String, Integer> model = new HashMap<>();
			Map<String, Integer> model_b = Train.arima(features, dataset, param, flavor_param(5), 1);
			Map<String, Integer> model_c = Train.recentMean(features, param, 6, true);
			
			for (String key : param.virtuals.keySet()) {
				double arima  = model_b.containsKey(key) ? model_b.get(key) : 0;
				double mean   = model_c.containsKey(key) ? model_c.get(key) : 0;
				
				int sum = (int) Math.ceil(( arima * 1 / 2 + mean * 1 / 2));
				model.put(key, sum >= 0 ? sum : 0);
			}
			return model;
		}
		else { // 默认加权平均
			Map<String, Integer> model = new HashMap<>();
			Map<String, Integer> model_b = Train.arima(features, dataset, param, flavor_param(5), 1);
			Map<String, Integer> model_a = Train.batchDiffRollLinearRegression(features, dataset, param, linear_diff, roll_param);
			Map<String, Integer> model_c = Train.recentMean(features, param, 5, true);
			
			for (String key : param.virtuals.keySet()) {
				double linear = model_a.containsKey(key) ? model_a.get(key) : 0;
				double arima  = model_b.containsKey(key) ? model_b.get(key) : 0;
				double mean   = model_c.containsKey(key) ? model_c.get(key) : 0;
				
				int sum = (int) Math.ceil((linear * 1 / 3 + arima * 1 / 3 + mean * 1 / 3));
				model.put(key, sum >= 0 ? sum : 0);
			}
			return model;
		}
	}
	
	@Deprecated
	public static Param inputParam(String[] inputContent) {
		Param params = new Param();
		
		// 1. 解析 总物理核数，内存， 硬盘大小
		String[] env_info = inputContent[0].split("\\s+");
		CPU cpu_all = new CPU(Integer.parseInt(env_info[0]), Integer.parseInt(env_info[1]), Integer.parseInt(env_info[2]));
		params.physics = cpu_all;
		
		// 2. 解析 虚拟机规格 参数
		Map<String, CPU> virtuals = new HashMap<>();
		int _index = 1;
		int _len   = inputContent.length;
		while(true) {
			while (_index < _len && inputContent[_index].equals("")) _index++;
			int num = Integer.parseInt(inputContent[_index++]);
			for (int i = 0; i < num; ++i, _index++) {
				String[] _info = inputContent[_index].split("\\s+");
				virtuals.put(_info[0], new CPU(Integer.parseInt(_info[1]), Integer.parseInt(_info[2])));
			}
			break;
		}
		params.virtuals = virtuals;
		
		// 3. 解析 优化目标
		
		while(true) {
			while (_index < _len && inputContent[_index].equals("")) _index++;
			params.target = inputContent[_index++];
			break;
		}
		
		// 4. 解析 预测时间段
		while(true) {
			while (_index < _len && inputContent[_index].equals("")) _index++;
			params.startTime = inputContent[_index++];
			params.endTime   = inputContent[_index++];
			break;
		}
		
		// 5. 解析间隔时间段
		params.span = DateUtil.span(params.startTime, params.endTime);
		
		//6. 加载holiday set
		params.holidaySet = holidaySet();
		
		//7. 加载long holiday set
		params.longHolidaySet = longHolidaySet();
		params.spring = longHolidaySet(spring);
		params.national = longHolidaySet(national);
		params.holiday_end = longHolidaySet(holidays_end);
		params.holiday_start = longHolidaySet(holidays_start);
		
		
		return params;
	}
	
	static int notGap = -1;
	public static Param mockParam(Param param, long endDate) {
		String startDate = DateUtil.long2Date((endDate + 2) * 1000 * 24 * 3600);
		long endTime = DateUtil.year2dayLong(param.startTime);
		String endDay = DateUtil.long2Date((endTime) * 1000 * 24 * 3600);
		
		notGap = (int) (endTime - endDate - 1);
		
		Param mockParam = new Param();
		mockParam.physicalCpusMap = param.physicalCpusMap;
		mockParam.virtuals = param.virtuals;
		mockParam.startTime = startDate;
		mockParam.endTime = endDay;
		
		mockParam.span = DateUtil.span(mockParam.startTime, mockParam.endTime);
		
		mockParam.holidaySet = holidaySet();
		mockParam.longHolidaySet = longHolidaySet();
		mockParam.spring = longHolidaySet(spring);
		mockParam.national = longHolidaySet(national);
		mockParam.holiday_end = longHolidaySet(holidays_end);
		mockParam.holiday_start = longHolidaySet(holidays_start);
		
		return mockParam;
	}
	
	public static Param inputMoreParam(String[] inputContent) {
		Param params = new Param();
		
		// 1. 解析 总物理核数，内存， 硬盘大小
		int _index = 0;
		int _num = Integer.parseInt(inputContent[_index++]);
		Map<String, CPU> cpusMap = new HashMap<>();
		
		for (int i = 0; i < _num; ++i) {
			String[] env_info = inputContent[i + 1].split("\\s+");
			CPU cpu = new CPU(env_info[0], Integer.parseInt(env_info[1]), Integer.parseInt(env_info[2]), Integer.parseInt(env_info[3]));
			cpusMap.put(env_info[0], cpu);
			_index ++;
		}
		params.physicalCpusMap = cpusMap;
		
		// 2. 解析 虚拟机规格 参数
		Map<String, CPU> virtuals = new HashMap<>();
		int _len   = inputContent.length;
		while(true) {
			while (_index < _len && inputContent[_index].equals("")) _index++;
			int num = Integer.parseInt(inputContent[_index++]);
			for (int i = 0; i < num; ++i, _index++) {
				String[] _info = inputContent[_index].split("\\s+");
				virtuals.put(_info[0], new CPU(Integer.parseInt(_info[1]), Integer.parseInt(_info[2])));
			}
			break;
		}
		params.virtuals = virtuals;
		
		// 3. 解析 预测时间段
		while(true) {
			while (_index < _len && inputContent[_index].equals("")) _index++;
			params.startTime = inputContent[_index++];
			params.endTime   = inputContent[_index++];
			break;
		}
		
		// 4. 解析间隔时间段
		params.span = DateUtil.span(params.startTime, params.endTime);
		
		// 5. 加载holiday set
		params.holidaySet = holidaySet();
		
		// 6. 加载long holiday set
		params.longHolidaySet = longHolidaySet();
		params.spring = longHolidaySet(spring);
		params.national = longHolidaySet(national);
		params.holiday_end = longHolidaySet(holidays_end);
		params.holiday_start = longHolidaySet(holidays_start);
		
		
		return params;
	}
	
	public static int fix(String key, int sum) {
		if (key.equals("flavor4")) return 0;
		return sum;
	}
	
	public static int seed = 2018;
	
	public static Map<String, Integer> avgFuse(Map<String, Integer>... models){
		int n = models.length;
		Map<String, Integer> model = new HashMap<>();
		for (int i = 0; i < n; ++i) {
			Map<String, Integer> m = models[i];
			for (String key : m.keySet()) {
				model.put(key, model.getOrDefault(key, 0) + m.get(key));
			}
		}
		
		for (String key : model.keySet()) {
			model.put(key, model.get(key) / n);
		}
		return model;
	}
	
	public static String[] predictVm(String[] ecsContent, String[] inputContent) {
		/** ===============定义输出=============== **/
		List<String> results = new ArrayList<String>();
		
		/** ===============加载训练数据集=============== **/
		Param params = inputMoreParam(inputContent);
		List<Record> dataset = LoadData.loadData(ecsContent);
		List<Feature> features = FeatureExtract.groupby(dataset, params, true, 4); // 取n天均值
		
		// 数据去噪
		features = FeatureExtract.denoise(features);
		
		// 需要判断一波 是否和之前的模型一致  !!!! 不能删除
		Param mockParam = mockParam(params, features.get(features.size() - 1).year2Day);
		
		if (notGap != 0) {
			List<Feature> lwlr = Train.localWeightedLR(features, mockParam, 3.5); // 抗干扰能力要比多项式强
			features.addAll(lwlr);
		}
		
		double fixed = Math.pow(1.31, Math.log((notGap + params.span) * 1.0 / 7 + 1)); //  必须从1.11往上开始调参
		Map<String, Integer> model = Train.recentDecreseMean(features, params, 15, 0.966, 0.66, 0.34, fixed); // alpha, m1, m2, fixed // 做统计的时候，可以考虑去掉节假日的flavor
		
		/** ===============备用模型=============== **/
//		Map<String, Integer> model = Train.recentMean(features, params, 15, 0.966);
//		double fixed = Math.pow(1.21, (notGap + params.span) * 1.0 / 7 + 1);
//		model = Train.poly_fixed(model, fixed);
		
		/** ===============使用trick=============== **/
//		Map<String, Integer> model = Train.trainMock(seed);
		model = Train.trick(model);
		model = Train.add(model, (int)(params.span * 1.0 / 7 + 0.5));
		
		
		
		/** ===============是该好好配置了=============== **/
		
		// 动态规划 + 时间段的预测（test 测试机的虚拟机数量）
		
		
		// 约束条件： 1. 虚拟机的核数 不超过 总核数 2. 虚拟机的内存 不超过总内存 
		
		List<Goods> goods = Allocation.transfer(model, params.virtuals);
		Map<String, Integer> other = new HashMap<>();
		for (String flavor : params.virtuals.keySet()) {
			other.put(flavor, 0);
		}
		
		Map<String, Integer> some = new HashMap<>();
		for (String flavor : params.virtuals.keySet()) {
			if (model.getOrDefault(flavor, 0) > 0) some.put(flavor, 20);
		}
		
		Map<String, List<Manifest>> answer = Allocation.allocateCPUAndMEM(params, goods, some, other);
		for (String flavor : other.keySet()) {
			model.put(flavor, model.getOrDefault(flavor, 0) + other.get(flavor));
		}
		
		int sum = 0;
		List<String> virtual_answer = new ArrayList<>();
		Map<String, Integer> virtual_map = new HashMap<>(); // int[] flavor int mem, int hhd, int cnt
		
		for (String key : params.virtuals.keySet()) {
			if (model != null && model.containsKey(key)) {
				int cnt = model.get(key);
				sum += cnt;
				
				virtual_answer.add(key + " " + cnt);
				virtual_map.put(key, cnt);
			}
			else {
				virtual_answer.add(key + " " + 0);
			}
		}
		
		results.add(sum +"");
		results.addAll(virtual_answer);
		
		for (String type : answer.keySet()) {
			results.add("");
			List<Manifest> manifests = answer.get(type);
			results.add(type + " " + manifests.size());
			
			for (Manifest menu : manifests) {
				StringBuilder sb = new StringBuilder();
				sb.append(type + "-" + menu.uuid + " ");
				Map<String, Integer> list = menu.getList();
				
				for (String key : list.keySet()) {
					sb.append(key + " " + list.get(key) + " ");
				}
				results.add(sb.deleteCharAt(sb.length() - 1).toString());
			}
		}
		return results.toArray(new String[0]);
	}

}
