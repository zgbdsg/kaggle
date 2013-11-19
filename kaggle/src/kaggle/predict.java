package kaggle;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.*;
import weka.core.converters.ConverterUtils.DataSource;

public class predict{
	/**
	 * @param args
	 */
	public static void main(String[] args){
		//Dataset train = new Dataset("C:\\Users\\zgb\\Desktop\\kaggle\\count.csv");
		//Dataset test = new Dataset("C:\\Users\\zgb\\Desktop\\kaggle\\count_test.csv");
		
		
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter("C:\\Users\\zgb\\Desktop\\kaggle\\answer.csv"));
			writer.write("QuestionId,IsTrue\n");
			DataSource trains = new DataSource("C:\\Users\\zgb\\Desktop\\kaggle\\count_train.arff");
			Instances trainins = trains.getDataSet();
			trainins.setClassIndex(trainins.numAttributes()-1);
			System.out.println(trainins.toString());
			DataSource tests = new DataSource("C:\\Users\\zgb\\Desktop\\kaggle\\count_test.arff");
			
			Instances testins = tests.getDataSet();
			//System.out.println(trainins.toString());
			
			Attribute label = trainins.attribute(trainins.classIndex());
			int[] lab = new int[1038];
			Enumeration en = label.enumerateValues();
			int index = 0;
			while(en.hasMoreElements()){
				Object elem = en.nextElement();
				lab[Integer.parseInt((String)elem)] = index;
				index ++;
			}
			
			BufferedReader read = new BufferedReader(new FileReader("C:\\Users\\zgb\\Desktop\\kaggle\\questions.csv"));
			read.readLine();
			
			String line;
			int numInstnaces = 0;
            while (read.readLine() != null) {
                numInstnaces++;
            }
            
            int[] testdevice = new int[numInstnaces];
            read = new BufferedReader(new FileReader("C:\\Users\\zgb\\Desktop\\kaggle\\questions.csv"));
            read.readLine();
            index = 0;
			while((line = read.readLine()) != null){
				String[] tmp = line.split(",");
				testdevice[index] = Integer.parseInt(tmp[2]);
				index ++;
			}
			
			Classifier nb = new NaiveBayes();
			nb.buildClassifier(trainins);
			for(int j=0;j < testins.numInstances();j ++){
				double[] result = nb.distributionForInstance(testins.get(j));
/*				for(int i=0;i<result.length;i ++)
					if(result[i] > 0.1)
						System.out.println(result[i]);*/
				
				writer.write(j+","+result[lab[testdevice[j]]] + "\n");
			}
			writer.close();
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
