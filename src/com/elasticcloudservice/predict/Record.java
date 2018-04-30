package com.elasticcloudservice.predict;

import com.date.utils.DateUtil;

public class Record{
	
	public String uuid;
	public String tag;
	public String date;
	
	// 1. 时间特征
	public long year2Day;
	
	// 2. week特征
	public int dayOfWeek;
	
	public Record(String uuid, String tag, String date){
		this.uuid = uuid;
		this.tag  = tag;
		this.date = date;
		
		String time = date.split("\\s+")[0];
		this.year2Day = DateUtil.year2dayLong(time);
		this.dayOfWeek = DateUtil.week(time);
			
	}

	@Override
	public String toString() {
		return "Record [uuid=" + uuid + ", tag=" + tag + ", date=" + date + ", year2Day=" + year2Day + ", dayOfWeek="
				+ dayOfWeek + "]";
	}
}
