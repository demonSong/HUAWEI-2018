package com.date.utils;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.elasticcloudservice.predict.Record;

/**
 * 
 * @author demonSong
 * @since 2018-03-12 15:54
 *
 */
public abstract class DateUtil {
	
	public static class Date{
		public int year;
		public int month;
		public int day;
		public int hour;
		public int minute;
		public int second;
		
		String year2Day;
		
		Date(String timestamp){
			String[] part = timestamp.split("\\s+");
			String[] part_1 = part[0].split("-");
			String[] part_2 = part[1].split(":");
			
			this.year2Day = part[0];
			
			this.year  = Integer.parseInt(part_1[0]);
			this.month = Integer.parseInt(part_1[1]);
			this.day   = Integer.parseInt(part_1[2]);
			 
			this.hour   = Integer.parseInt(part_2[0]);
			this.minute = Integer.parseInt(part_2[1]);
			this.second = Integer.parseInt(part_2[2]);
		}
		
		String parseYear2Day() {
			return this.year2Day;
		}
		
		int parseDay(){
			return this.day;
		}

		@Override
		public String toString() {
			return "Date [year=" + year + ", month=" + month + ", day=" + day + ", hour=" + hour + ", minute=" + minute
					+ ", second=" + second + "]";
		}
	}
	
	private static List<Date> transfer(List<Record> dataset){
		List<Date> dates = new ArrayList<>();
		for (Record instance : dataset) {
			dates.add(new Date(instance.date));
		}
		return dates;
	}
	
	/**
	 * 1 -> 周日
	 * 7 -> 周六
	 * @param time
	 * @return
	 * @throws ParseException
	 */
	public static int week(String time) {
		try {
			SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
			java.util.Date t = format.parse(time);
			Calendar ca = Calendar.getInstance();
			ca.setTime(t);
			return ca.get(Calendar.DAY_OF_WEEK);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	public static int month(String time){
		return new Date(time).month;
	}
	
	public static int dayOfMonth(String time){
		return new Date(time).day;
	}
	
	public static int weekOfMonth(String time){
		try {
			SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
			java.util.Date t = format.parse(time);
			Calendar ca = Calendar.getInstance();
			ca.setTime(t);
			return ca.get(Calendar.WEEK_OF_MONTH);
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	/**
	 * 一年中的第几周
	 * @param time
	 * @return
	 */
	public static int weekOfYear(String time){
		try {
			SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
			java.util.Date t = format.parse(time);
			Calendar ca = Calendar.getInstance();
			ca.setTime(t);
			return ca.get(Calendar.WEEK_OF_YEAR);
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	
	public static int long2week(long timestamp) {
		Calendar ca = Calendar.getInstance();
		ca.setTimeInMillis(timestamp);
		return ca.get(Calendar.DAY_OF_WEEK);
	}
	
	public static String long2Date(long timestamp) {
		SimpleDateFormat sf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");  
		java.util.Date date = new java.util.Date(timestamp);  
	    return sf.format(date);  
	}
	
	public static long year2dayLong(String time){
		try {
			SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
			java.util.Date t = format.parse(time);
			return t.getTime() / (24 * 3600 * 1000);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	public static long year2dayLong(String time, SimpleDateFormat format) throws ParseException {
		java.util.Date t = format.parse(time);
		return t.getTime() / (24 * 3600 * 1000);
	}
	
	
	public static int totalDiffDayInDataset(List<Record> dataset) {
		List<Date> dates = DateUtil.transfer(dataset);
		Set<String> diff = new HashSet<>();
		for (Date date : dates) diff.add(date.parseYear2Day());
		return diff.size();
	}
	
	public static int span(String startTime, String endTime) {
		Date start = new Date(startTime);
		Date end   = new Date(endTime);
		
		SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd");
		java.util.Date s_;
		java.util.Date e_;
		
		try {
			s_ = format.parse(start.parseYear2Day());
			e_ = format.parse(end.parseYear2Day());
			return (int) ((e_.getTime() - s_.getTime()) / (1000 * 3600 * 24));
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return 7;
	}
	
	public static void main(String[] args) throws ParseException {
	}
}
