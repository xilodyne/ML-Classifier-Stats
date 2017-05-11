package xilodyne.machinelearning.test.datumbox;

//be sure to point eclipse to the resoure folder at
//data/datumbox-framework-examples-0.8.0_resources
//
//add lib:  logback-core-1.1.9.jar
import com.datumbox.framework.common.Configuration;
import com.datumbox.framework.core.common.dataobjects.Dataframe;
import com.datumbox.framework.core.common.dataobjects.Record;
import com.datumbox.framework.common.dataobjects.TypeInference;
import com.datumbox.framework.common.utilities.RandomGenerator;
import com.datumbox.framework.core.machinelearning.MLBuilder;
import com.datumbox.framework.core.machinelearning.classification.*;
import com.datumbox.framework.core.machinelearning.clustering.Kmeans;
import com.datumbox.framework.core.machinelearning.featureselection.PCA;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.ClassificationMetrics;
import com.datumbox.framework.core.machinelearning.modelselection.metrics.ClusteringMetrics;
import com.datumbox.framework.core.machinelearning.modelselection.splitters.ShuffleSplitter;
import com.datumbox.framework.core.machinelearning.preprocessing.OneHotEncoder;
import com.datumbox.framework.core.machinelearning.preprocessing.MinMaxScaler;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.LoggerCSV;
import xilodyne.util.metrics.OutputResults;
import xilodyne.util.metrics.TestResultsDataML;


/**
 * Copyright (C) 2013-2017 Vasilis Vryniotis <bbriniotis@datumbox.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//package com.datumbox.examples;

/**
 * Clustering example.
 * 
 * @author Vasilis Vryniotis <bbriniotis@datumbox.com>
 */
public class Classification_IrisData_MultinominalNB {
    
    /**
     * Example of how to use directly the algorithms of the framework in order to
     * perform clustering. A similar approach can be used to perform classification,
     * regression, build recommender system or perform topic modeling and dimensionality
     * reduction.
     * 
     * @param args the command line arguments
     */
	
	public static String CSV_Filename = "IRIS_Data_NaiveBayes-datumbox";
	public static String delimiter = ",";
	public static String[] header = {"timestamp", "class name", 
		"accuracy", "CV Fold", "# of lines", "# trained", "date time", 
		"train time", "predict time","total time"};

	private static Logger log;
	private static LoggerCSV logCSV;
	
    public static void main(String[] args) {  
        /**
         * There are 5 configuration files in the resources folder:
         *
         * - datumbox.configuration.properties: It defines for the default storage engine (required)
         * - datumbox.concurrencyconfiguration.properties: It controls the concurrency levels (required)
         * - datumbox.inmemoryconfiguration.properties: It contains the configurations for the InMemory storage engine (required)
         * - datumbox.mapdbconfiguration.properties: It contains the configurations for the MapDB storage engine (optional)
         * - logback.xml: It contains the configuration file for the logger (optional)
         */    
    	
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		 G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);

		String className = "com.datumbox.framework.core.machinelearning.classification." +
				//"MultinomialNaiveBayes";
		        //"SoftMaxRegression";
		        //"BernoulliNaiveBayes";
		        "BinarizedNaiveBayes";
		        //"MaximumEntropy";
		        //"OrdinalRegression";
		        //"SupportVectorMachine";

		TestResultsDataML resultsData = new TestResultsDataML();
		resultsData.setClassMLName(className);

//		log = new Logger("logs", "IRIS_xd_GNB_NoCV" + "_" + className.substring(className.lastIndexOf(".") + 1));
		log = new Logger("results", "IRIS-datumbox-NB-CV" + "_" + resultsData.getClassMLNameWithoutDomain());

		logCSV = new LoggerCSV("results", CSV_Filename, 
				delimiter, header);
		logCSV.log_CSV_Timestamp();
		logCSV.log_CSV_Entry(resultsData.getClassMLName());

		
		log.logln_withClassName(G.lF,"");

		resultsData.setStartData();
        
        //Initialization
        //--------------
        RandomGenerator.setGlobalSeed(42L); //optionally set a specific seed for all Random objects
        Configuration configuration = Configuration.getConfiguration(); //default configuration based on properties file
        //configuration.setStorageConfiguration(new InMemoryConfiguration()); //use In-Memory engine (default)
        //configuration.setStorageConfiguration(new MapDBConfiguration()); //use MapDB engine
        //configuration.getConcurrencyConfiguration().setParallelized(true); //turn on/off the parallelization
        //configuration.getConcurrencyConfiguration().setMaxNumberOfThreadsPerTask(4); //set the concurrency level
        
        
        
        //Reading Data
        //------------
        Dataframe data;
        try (Reader fileReader = new InputStreamReader(new FileInputStream(Paths.get(Classification_IrisData_MultinominalNB.class.getClassLoader().getResource("datasets/uci-iris/iris.csv").toURI()).toFile()), "UTF-8")) {
            LinkedHashMap<String, TypeInference.DataType> headerDataTypes = new LinkedHashMap<>();
            headerDataTypes.put("sepal length", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("sepal width", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("petal length", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("petal width", TypeInference.DataType.NUMERICAL);
            headerDataTypes.put("class", TypeInference.DataType.CATEGORICAL);
            
            data = Dataframe.Builder.parseCSVFile(fileReader, "class", headerDataTypes, ',', '"', "\r\n", null, null, configuration);
        }
        catch(UncheckedIOException | IOException | URISyntaxException ex) {
            throw new RuntimeException(ex);
        }

        //Spit into train and test datasets
 //       ShuffleSplitter.Split split = new ShuffleSplitter(0.8, 1).split(data).next();
        ShuffleSplitter.Split split = new ShuffleSplitter(0.9, 1).split(data).next();
        Dataframe trainingDataframe = split.getTrain();
        Dataframe testingDataframe = split.getTest();
        
        Iterator dataval = trainingDataframe.iterator();
        
        while (dataval.hasNext()) {
        	System.out.println("value: " + dataval.next());
        }
        
        resultsData.setEndData();
        
 /*       //Transform Dataframe
        //-----------------
        
        //Scale continuous variables
        MinMaxScaler.TrainingParameters nsParams = new MinMaxScaler.TrainingParameters();
        MinMaxScaler numericalScaler = MLBuilder.create(nsParams, configuration);

        numericalScaler.fit_transform(trainingDataframe);
        numericalScaler.save("Iris");
        


        //Feature Selection
        //-----------------
        
        //Perform dimensionality reduction using PCA

        PCA.TrainingParameters featureSelectionParameters = new PCA.TrainingParameters();
        featureSelectionParameters.setMaxDimensions(trainingDataframe.xColumnSize()-1); //remove one dimension
        featureSelectionParameters.setWhitened(false);
        featureSelectionParameters.setVariancePercentageThreshold(0.99999995);

        PCA featureSelection = MLBuilder.create(featureSelectionParameters, configuration);
        featureSelection.fit_transform(trainingDataframe);
        featureSelection.save("Iris");
        
        
       */ 
        resultsData.setStartFit();
        //Fit the classifier
        //------------------

        
         //MultinomialNaiveBayes classifier;
        //SoftMaxRegression classifier;
        //BernoulliNaiveBayes classifier;
        BinarizedNaiveBayes classifier;
        //MaximumEntropy classifier;
        //OrdinalRegression classifier;
        //SupportVectorMachine classifier;

        
        //MultinomialNaiveBayes.TrainingParameters param = new MultinomialNaiveBayes.TrainingParameters();
//        SoftMaxRegression.TrainingParameters param = new SoftMaxRegression.TrainingParameters();
        //BernoulliNaiveBayes.TrainingParameters param = new BernoulliNaiveBayes.TrainingParameters();
        BinarizedNaiveBayes.TrainingParameters param = new BinarizedNaiveBayes.TrainingParameters();
//        MaximumEntropy.TrainingParameters param = new MaximumEntropy.TrainingParameters();
//        OrdinalRegression.TrainingParameters param = new OrdinalRegression.TrainingParameters();
//        SupportVectorMachine.TrainingParameters param = new SupportVectorMachine.TrainingParameters();
//        param.setTotalIterations(200);
//        param.setLearningRate(0.1);

        classifier = MLBuilder.create(param, configuration);
//        SoftMaxRegression classifier = MLBuilder.create(param, configuration);
        classifier.fit(trainingDataframe);
        classifier.save("Iris");
        
        resultsData.setEndFit();
        //Use the classifier
        //------------------
        
        //Apply the same numerical scaling on testingDataframe
//        numericalScaler.transform(testingDataframe);
        
        //Apply the same featureSelection transformations on testingDataframe
 //       featureSelection.transform(testingDataframe);

        resultsData.setStartPredict();
        //Use the classifier to make predictions on the testingDataframe
        classifier.predict(testingDataframe);
        resultsData.setEndPredict();
        
        //Get validation metrics on the test set
        ClassificationMetrics vm = new ClassificationMetrics(testingDataframe);
        
        System.out.println("Results:");
        for(Map.Entry<Integer, Record> entry: testingDataframe.entries()) {
            Integer rId = entry.getKey();
            Record r = entry.getValue();
            System.out.println("Record "+rId+" - Real Y: "+r.getY()+", Predicted Y: "+r.getYPredicted());
        }
        
        System.out.println("Classifier Accuracy: "+vm.getAccuracy());
        
        
		resultsData.setAccuracy(vm.getAccuracy());
		resultsData.setTrainingDataSize(trainingDataframe.size());
		resultsData.setTestingDataSize(testingDataframe.size());
		OutputResults.getMLStats(log, logCSV, resultsData);
		logCSV.log_CSV_EOL();

        
        //Clean up
        //--------
        
        //Delete scaler, featureselector and classifier.
 //       numericalScaler.delete();
 //       featureSelection.delete();
        classifier.delete();
        
        //Close Dataframes.
        trainingDataframe.close();
        testingDataframe.close();
    }
}