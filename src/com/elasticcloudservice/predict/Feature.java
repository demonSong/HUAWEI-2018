package com.elasticcloudservice.predict;

public class Feature {
	
	public long year2Day;
	public int dayOfWeek;
	public String tag;
	public int count;
	
	public boolean isHoliday;
	
	public String date;
	
	public Feature(){
		
	}
	
	
	@Override
	public String toString() {
		return "Feature [year2Day=" + year2Day + ", dayOfWeek=" + dayOfWeek + ", tag=" + tag + ", count=" + count + "]";
	}
}
