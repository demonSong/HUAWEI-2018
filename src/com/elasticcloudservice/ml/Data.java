//parse the csv file, get features and label. the format: feature1,feature2,...,label
//first scan, get the feature dimension, dataset size, count of missing value for each feature
//second scan, get each feature's (value,index) and missing value indexes
//if we use ArrayList,only one scanning is needed, but it is memory consumption

package com.elasticcloudservice.ml;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Data{
    public static float NULL = -Float.MAX_VALUE;
}

class TrainData extends Data{
    public float[][][] feature_value_index;
    public double[] label;
    public int[][] missing_index;
    public int feature_dim;
    public int dataset_size;
    private ArrayList<Integer> missing_count = new ArrayList<>();
    public float[][] origin_feature;
    private ArrayList<String> cat_features_names;
    public ArrayList<Integer> cat_features_cols = new ArrayList<>();

    public TrainData(List<Instance> instances) {
    	this.cat_features_names = new ArrayList<>();
    	first_scan(instances);
    	second_scan(instances);
    }
    
    public TrainData(String file, ArrayList<String> categorical_features){
        this.cat_features_names = categorical_features;
        first_scan(file);
        second_scan(file);
    }
    
    private void first_scan(List<Instance> instances){
        feature_dim = instances.get(0).getFeature_dim();
        
        for (int i = 0; i < feature_dim; ++i) {
        	cat_features_names.add("feature_" + i);
        }
        
		for (int i = 0; i < feature_dim; ++i) {
			cat_features_cols.add(i);
		}

		for (int i = 0; i < feature_dim; ++i) {
			missing_count.add(0);
		}
		
		this.dataset_size = instances.size();
    }
    
    private void second_scan(List<Instance> instances){
    	label = new double[dataset_size];
        origin_feature = new float[dataset_size][feature_dim];
        missing_index = new int[feature_dim][];
        feature_value_index = new float[feature_dim][][];

		for (int i = 0; i < feature_dim; i++) {
            int cnt = missing_count.get(i);
            missing_index[i] = new int[cnt];
            feature_value_index[i] = new float[dataset_size-cnt][2];
        }

        int[] cur_index = new int[feature_dim];
        int[] cur_missing_index = new int[feature_dim];
        
        Arrays.fill(cur_index, 0);
        Arrays.fill(cur_missing_index, 0);
        
		for (int row = 0; row < dataset_size; ++row) {
            Instance instance = instances.get(row);
            label[row] = instance.y;

			for (int col = 0; col < feature_dim; ++col) {
                feature_value_index[col][cur_index[col]][0] = (float) instance.X[col];
                feature_value_index[col][cur_index[col]][1] = row;
                cur_index[col] += 1;
                origin_feature[row][col] = (float) instance.X[col];
			}
        }
    }

    //to obtain: feature_dim, dataset_size,missing_count,cat_features_dim
    private void first_scan(String file){
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            String[] columns = header.split(",");
            feature_dim = columns.length - 1;

            for(int i=0;i<columns.length;i++){
                if(cat_features_names.contains(columns[i])){
                    cat_features_cols.add(i);
                }
            }

            for(int i=0;i<feature_dim;i++){
                missing_count.add(0);
            }

            String line;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                String[] strs = line.split(",");
                dataset_size += 1;
                for(int i=0;i<feature_dim;i++){
                    if(strs[i].equals("")){
                        missing_count.set(i,missing_count.get(i)+1);
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //to obtain:feature_value_index,label,missing_index,origin_feature,cat_features_values
    private void second_scan(String file){
        label = new double[dataset_size];
        missing_index = new int[feature_dim][];
        feature_value_index = new float[feature_dim][][];

        for(int i=0;i<feature_dim;i++){
            int cnt = missing_count.get(i);
            missing_index[i] = new int[cnt];
            feature_value_index[i] = new float[dataset_size-cnt][2];
        }

        origin_feature = new float[dataset_size][feature_dim];

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();

            int[] cur_index = new int[feature_dim];
            int[] cur_missing_index = new int[feature_dim];
            Arrays.fill(cur_index,0);
            Arrays.fill(cur_missing_index,0);

            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                label[row] = Float.parseFloat(strs[strs.length-1]);

                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        missing_index[col][cur_missing_index[col]] = row;
                        cur_missing_index[col] += 1;
                        origin_feature[row][col] = Data.NULL;
                    }else{
                        feature_value_index[col][cur_index[col]][0] = Float.parseFloat(strs[col]);
                        feature_value_index[col][cur_index[col]][1] = row;
                        cur_index[col] += 1;
                        origin_feature[row][col] = Float.parseFloat(strs[col]);
                    }
                }
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }
}


class ValidationData extends Data{
    public int feature_dim;
    public int dataset_size;
    public float[][] origin_feature;
    public double[] label;
    
    public ValidationData(List<Instance> valid){
        first_scan(valid);
        second_scan(valid);
    }
    
    public ValidationData(String file){
        first_scan(file);
        second_scan(file);
    }
    
    private void first_scan(List<Instance> valid){
        feature_dim = valid.get(0).getFeature_dim();
        dataset_size = valid.size();
    }
    
    private void second_scan(List<Instance> valid){
        label = new double[dataset_size];
        origin_feature = new float[dataset_size][feature_dim];
		for (int row = 0; row < dataset_size; ++row) {
			Instance in = valid.get(row);
            label[row] = in.y;
			for (int col = 0; col < feature_dim; ++col) {
				origin_feature[row][col] = (float) in.X[col];
            }
        }
    }
    
    private void first_scan(String file){
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            feature_dim = header.split(",").length - 1;

            String line;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                dataset_size += 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void second_scan(String file){
        label = new double[dataset_size];
        origin_feature = new float[dataset_size][feature_dim];

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();
            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                label[row] = Float.parseFloat(strs[strs.length-1]);
                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        origin_feature[row][col] = Data.NULL;
                    }else{
                        origin_feature[row][col] = Float.parseFloat(strs[col]);
                    }
                }
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }

}


class TestData extends Data{
    public int feature_dim;
    public int dataset_size;
    public float[][] origin_feature;

    public TestData(String file){
        first_scan(file);
        second_scan(file);
    }

    private void first_scan(String file){
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            feature_dim = header.split(",").length;

            String line;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                dataset_size += 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void second_scan(String file){
        origin_feature = new float[dataset_size][feature_dim];

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();
            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        origin_feature[row][col] = Data.NULL;
                    }else{
                        origin_feature[row][col] = Float.parseFloat(strs[col]);
                    }
                }
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }

}
