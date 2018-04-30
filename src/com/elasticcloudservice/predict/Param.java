package com.elasticcloudservice.predict;

import java.util.Map;
import java.util.Set;

public class Param {
	
	
	@Deprecated
	public CPU physics;   // 物理机器信息
	
	public Map<String, CPU> physicalCpusMap;
	
	public String target; // 优化目标
	
	public String startTime; // 预测开始时间
	public String endTime;   // 预测结束时间
	
	public Map<String, CPU> virtuals; // 虚拟机
	
	public Set<String> holidaySet;
	public Set<Long> longHolidaySet;
	
	public Set<Long> spring;
	public Set<Long> national;
	public Set<Long> holiday_end;
	public Set<Long> holiday_start;
	
	public int span; // 时间间隔
	
	public Param() {
	}
}
