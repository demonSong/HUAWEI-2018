package com.elasticcloudservice.ml;

import java.util.ArrayList;
import java.util.List;

public class Xgboost {
	
	List<Instance> dataset;
	List<Instance> train;
	List<Instance> valid;
	
	public Xgboost(List<Instance> dataset, boolean useValid, double ratio) {
		this.dataset = dataset;
		this.train = new ArrayList<>();
		this.valid = new ArrayList<>();
		if (useValid) validation_split(ratio);
	}
	
	/**
	 * 注意时间序列请保持原有顺序
	 * @param ratio
	 */
	public void validation_split(double ratio) {
		int size = dataset.size();
		int train_size = size - (int) (size * ratio);
		
		for (int i = 0; i < size; ++i) {
			Instance data = dataset.get(i);
			if (i < train_size) train.add(data);
			else valid.add(data);
		}
	}
	
    public GBM training( int num_boost_round,
    					 int early_stopping_round,
    					 boolean maximize,
    					 String eval_metric,
    					 String loss,
    					 double eta,
    					 int max_depth,
    					 double scale_pos_weight,
    					 double rowsample,
    					 double colsample,
    					 double min_child_weight,
    					 int min_sample_split,
    					 double lambda,
    					 double gamma,
    					 int num_thread){
    	
    	if (eval_metric.equals("mse")) maximize = false;
    	
        GBM val = new GBM();
        
        val.fit(this.train,
                this.valid,
                early_stopping_round,
                maximize,
                eval_metric,
                loss,
                eta,
                num_boost_round,
                max_depth,
                scale_pos_weight,
                rowsample,
                colsample,
                min_child_weight,
                min_sample_split,
                lambda,
                gamma,
                num_thread);
        
        GBM tgb = new GBM();
        tgb.fit(this.dataset,
                null,
                early_stopping_round,
                maximize,
                eval_metric,
                loss,
                eta,
                val.get_best_round(),
                max_depth,
                scale_pos_weight,
                rowsample,
                colsample,
                min_child_weight,
                min_sample_split,
                lambda,
                gamma,
                num_thread);
        
        return tgb;
    }

    public static void main(String[] args){
    	
    }
}
