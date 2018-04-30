package com.elasticcloudservice.predict;

import java.util.ArrayList;
import java.util.List;

public class LoadData {
	
	/**
	 * 根据 字符数组 加载训练数据集
	 * @param content
	 * @return
	 */
	public static List<Record> loadData(String[] content){
		List<Record> dataset = new ArrayList<>();
		
		for (int i = 0; i < content.length; i++) {
			if (content[i].contains(" ")
					&& content[i].split("\\s+").length == 4) {

				String[] array = content[i].split("\\s+");
				String uuid = array[0];
				String flavorName = array[1];
				String createTime = array[2];
				String hour = array[3];

				dataset.add(new Record(uuid, flavorName, createTime + " " + hour));
			}
		}
		return dataset;
	}
	
	public static Record loadData(String content) {
		String[] array = content.split("\\s+");
		String uuid = array[0];
		String flavorName = array[1];
		String createTime = array[2];
		String hour = array[3];
		return new Record(uuid, flavorName, createTime + " " + hour);
	}
}
