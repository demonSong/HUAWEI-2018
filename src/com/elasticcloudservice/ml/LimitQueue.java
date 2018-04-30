package com.elasticcloudservice.ml;

import java.util.LinkedList;
import java.util.Queue;

public class LimitQueue{
	
	private int limit;
	private Queue<Integer> queue;
	
	public LimitQueue(int limit) {
		this.limit = limit;
		queue = new LinkedList<>();
	}

	public boolean offer(Integer e) {
		if (queue.size() >= limit) {
			queue.poll();
		}
		return queue.offer(e);
	}
	
	public Integer peek() {
		return queue.peek();
	}

	public Integer poll() {
		return queue.poll();
	}
	
	public double getMean() {
		double sum = 0.0;
		Integer[] arra = queue.toArray(new Integer[0]);
		for (int val : arra) sum += val;
		return sum / arra.length;
	}
	
	public static void main(String[] args) {
		LimitQueue queue = new LimitQueue(3);
		for (int x = 1; x <= 10; ++x) {
			queue.offer(x);
			System.out.println(queue.getMean());
		}
	}
}
