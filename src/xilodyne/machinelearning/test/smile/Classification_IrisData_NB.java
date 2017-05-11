package xilodyne.machinelearning.test.smile;

import java.io.BufferedReader;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import smile.classification.NaiveBayes;
import static org.junit.Assert.*;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.feature.Bag;
import smile.math.Math;
import smile.stat.distribution.Distribution;
import smile.stat.distribution.GaussianMixture;
import smile.validation.CrossValidation;
import smile.validation.LOOCV;
import xilodyne.util.ArrayUtils;
import xilodyne.util.Logger;
import xilodyne.util.LoggerCSV;
import xilodyne.util.metrics.OutputResults;
import xilodyne.util.metrics.TestResultsDataML;

public class Classification_IrisData_NB {
	
	//change IOUtils to point to correct data directory
	   public static void main(String[] args) {  
		NaiveBayesTest nbTest = new NaiveBayesTest(); 
		
		try {
		
			//pre-written method for iris data
			nbTest.testPredict();
		} catch (Exception e) {

			e.printStackTrace();
		}
		
	   }
	   
	}

/**
*
* @author Haifeng Li
*/
 class NaiveBayesTest {

	  
   String[] feature = {
       "outstanding", "wonderfully", "wasted", "lame", "awful", "poorly",
       "ridiculous", "waste", "worst", "bland", "unfunny", "stupid", "dull",
       "fantastic", "laughable", "mess", "pointless", "terrific", "memorable",
       "superb", "boring", "badly", "subtle", "terrible", "excellent",
       "perfectly", "masterpiece", "realistic", "flaws"
   };
   double[][] moviex;
   int[] moviey;

   public NaiveBayesTest() {
	   System.out.println("NB Test init");
       String[][] x = new String[2000][];
       int[] y = new int[2000];

       try(BufferedReader input = smile.data.parser.IOUtils.getTestDataReader("text/movie.txt")) {
           for (int i = 0; i < x.length; i++) {
               String[] words = input.readLine().trim().split(" ");

               if (words[0].equalsIgnoreCase("pos")) {
                   y[i] = 1;
               } else if (words[0].equalsIgnoreCase("neg")) {
                   y[i] = 0;
               } else {
                   System.err.println("Invalid class label: " + words[0]);
               }

               x[i] = words;
           }
       } catch (IOException ex) {
           System.err.println(ex);
       }

       moviex = new double[x.length][];
       moviey = new int[y.length];
       Bag<String> bag = new Bag<>(feature);
       for (int i = 0; i < x.length; i++) {
           moviex[i] = bag.feature(x[i]);
           moviey[i] = y[i];
       }
   }


   @BeforeClass
   public static void setUpClass() throws Exception {
   }

   @AfterClass
   public static void tearDownClass() throws Exception {
   }

   @Before
   public void setUp() {
   }

   @After
   public void tearDown() {
   }

   /**
    * Test of predict method, of class NaiveBayes.
    */
   @Test
   public void testPredict() {
       System.out.println("predict");
   	String CSV_Filename = "IRIS_Data_NaiveBayes-smile";
   	String delimiter = ",";
   	String[] header = {"timestamp", "class name", 
   		"accuracy", "CV Fold", "# of lines", "# trained", "date time", 
   		"train time", "predict time","total time"};

   	//Logger log;
   	LoggerCSV logCSV;
	String className = "smile.classification.NaiveBayes";
	TestResultsDataML resultsData = new TestResultsDataML();
	resultsData.setClassMLName(className);
	
	logCSV = new LoggerCSV("results", CSV_Filename, 
			delimiter, header);
	logCSV.log_CSV_Timestamp();
	logCSV.log_CSV_Entry(resultsData.getClassMLName());



       
//		Instant startData, startEnd, startFit, endFit, startPredict, endPredict;

		//startFit = Instant.now();
	resultsData.setStartData();
       ArffParser arffParser = new ArffParser();
       arffParser.setResponseIndex(4);
       try {
    	 
    	   
           AttributeDataset iris = arffParser.parse(smile.data.parser.IOUtils.getTestDataFile("weka/iris.arff"));
           double[][] x = iris.toArray(new double[iris.size()][]);
           int[] y = iris.toArray(new int[iris.size()]);

           int n = x.length;
           
           System.out.println(ArrayUtils.printArray(x));
           System.out.println(ArrayUtils.printArray(y));
           System.out.println("x length: " + n);
           LOOCV loocv = new LOOCV(n);
           int error = 0;

           resultsData.setEndData();
           resultsData.setStartPredict();

           int[] trainy = null;
           for (int l = 0; l < n; l++) {
               double[][] trainx = Math.slice(x, loocv.train[l]);
               //int[] trainy = Math.slice(y, loocv.train[l]);
               trainy = Math.slice(y, loocv.train[l]);

               int p = trainx[0].length;
               int k = Math.max(trainy) + 1;

               
               double[] priori = new double[k];
               Distribution[][] condprob = new Distribution[k][p];
               for (int i = 0; i < k; i++) {
      //System.out.println("l:i -> k" + l +":"+ i +" -> "+ k);
                   priori[i] = 1.0 / k;
                   for (int j = 0; j < p; j++) {
                       ArrayList<Double> axi = new ArrayList<>();
                       for (int m = 0; m < trainx.length; m++) {
                           if (trainy[m] == i) {
                               axi.add(trainx[m][j]);
                           }
                       }

                       double[] xi = new double[axi.size()];
                       for (int m = 0; m < xi.length; m++) {
                           xi[m] = axi.get(m);
                       }

                       condprob[i][j] = new GaussianMixture(xi, 3);
                   }
               }
               NaiveBayes bayes = new NaiveBayes(priori, condprob);


               //System.out.println("l, loocv.test[l]:" + l +", " + loocv.test[l]);
               //l, loocv.test[l]:148, 148
               //l, loocv.test[l]:149, 149
               if (y[loocv.test[l]] != bayes.predict(x[loocv.test[l]]))
                   error++;
           }
           System.out.println("Error: " + error);
           System.out.println("y length: " + trainy.length + ", " + y.length);
           double accuracy = 100 - ((error / (double) y.length) * 100);
           System.out.println("error / trainy.length: " + accuracy);
           
           //endFit = Instant.now();
           resultsData.setEndPredict();
           
   		resultsData.setAccuracy(accuracy);
   		resultsData.setTrainingDataSize(0);
   		resultsData.setTestingDataSize(trainy.length);
   		OutputResults.getMLStats(null, logCSV, resultsData);
   		logCSV.log_CSV_EOL();

       
//			long trainingTime = Duration.between(startFit, endFit).toMillis() ;
			//long trainingTime = Duration.between(startFit, endFit).getSeconds();
//			double dtrainTime = trainingTime / (double) 1000;

  //         System.out.format("Iris error rate = %.2f%%%n", 100.0 * error / x.length);
  //         System.out.format("Iris accuracy rate = %.2f%%%n", (x.length - error) / (double) x.length);
//	System.out.println("Predict time GNB: " + dtrainTime + " seconds.");

  //         assertEquals(5, error);
       } catch (Exception ex) {
           System.err.println(ex);
       }

   }

   /**
    * Test of learn method, of class SequenceNaiveBayes.
    */
   @Test
   public void testLearnMultinomial() {
       System.out.println("batch learn Multinomial");

       double[][] x = moviex;
       int[] y = moviey;
       int n = x.length;
       int k = 10;
       CrossValidation cv = new CrossValidation(n, k);
       int error = 0;
       int total = 0;
       for (int i = 0; i < k; i++) {
           double[][] trainx = Math.slice(x, cv.train[i]);
           int[] trainy = Math.slice(y, cv.train[i]);
           NaiveBayes bayes = new NaiveBayes(NaiveBayes.Model.MULTINOMIAL, 2, feature.length);

           bayes.learn(trainx, trainy);

           double[][] testx = Math.slice(x, cv.test[i]);
           int[] testy = Math.slice(y, cv.test[i]);
           for (int j = 0; j < testx.length; j++) {
               int label = bayes.predict(testx[j]);
               if (label != -1) {
                   total++;
                   if (testy[j] != label) {
                       error++;
                   }
               }
           }
       }

       System.out.format("Multinomial error = %d of %d%n", error, total);
       assertTrue(error < 265);
   }

   /**
    * Test of learn method, of class SequenceNaiveBayes.
    */
   @Test
   public void testLearnMultinomial2() {
       System.out.println("online learn Multinomial");

       double[][] x = moviex;
       int[] y = moviey;
       int n = x.length;
       int k = 10;
       CrossValidation cv = new CrossValidation(n, k);
       int error = 0;
       int total = 0;
       for (int i = 0; i < k; i++) {
           double[][] trainx = Math.slice(x, cv.train[i]);
           int[] trainy = Math.slice(y, cv.train[i]);
           NaiveBayes bayes = new NaiveBayes(NaiveBayes.Model.MULTINOMIAL, 2, feature.length);

           for (int j = 0; j < trainx.length; j++) {
               bayes.learn(trainx[j], trainy[j]);
           }

           double[][] testx = Math.slice(x, cv.test[i]);
           int[] testy = Math.slice(y, cv.test[i]);
           for (int j = 0; j < testx.length; j++) {
               int label = bayes.predict(testx[j]);
               if (label != -1) {
                   total++;
                   if (testy[j] != label) {
                       error++;
                   }
               }
           }
       }

       System.out.format("Multinomial error = %d of %d%n", error, total);
       assertTrue(error < 265);
   }

   /**
    * Test of learn method, of class SequenceNaiveBayes.
    */
   @Test
   public void testLearnBernoulli() {
       System.out.println("batch learn Bernoulli");

       double[][] x = moviex;
       int[] y = moviey;
       int n = x.length;
       int k = 10;
       CrossValidation cv = new CrossValidation(n, k);
       int error = 0;
       int total = 0;
       for (int i = 0; i < k; i++) {
           double[][] trainx = Math.slice(x, cv.train[i]);
           int[] trainy = Math.slice(y, cv.train[i]);
           NaiveBayes bayes = new NaiveBayes(NaiveBayes.Model.BERNOULLI, 2, feature.length);

           bayes.learn(trainx, trainy);

           double[][] testx = Math.slice(x, cv.test[i]);
           int[] testy = Math.slice(y, cv.test[i]);

           for (int j = 0; j < testx.length; j++) {
               int label = bayes.predict(testx[j]);
               if (label != -1) {
                   total++;
                   if (testy[j] != label) {
                       error++;
                   }
               }
           }
       }

       System.out.format("Bernoulli error = %d of %d%n", error, total);
       assertTrue(error < 270);
   }

   /**
    * Test of learn method, of class SequenceNaiveBayes.
    */
   @Test
   public void testLearnBernoulli2() {
       System.out.println("online learn Bernoulli");

       double[][] x = moviex;
       int[] y = moviey;
       int n = x.length;
       int k = 10;
       CrossValidation cv = new CrossValidation(n, k);
       int error = 0;
       int total = 0;
       for (int i = 0; i < k; i++) {
           double[][] trainx = Math.slice(x, cv.train[i]);
           int[] trainy = Math.slice(y, cv.train[i]);
           NaiveBayes bayes = new NaiveBayes(NaiveBayes.Model.BERNOULLI, 2, feature.length);

           for (int j = 0; j < trainx.length; j++) {
               bayes.learn(trainx[j], trainy[j]);
           }

           double[][] testx = Math.slice(x, cv.test[i]);
           int[] testy = Math.slice(y, cv.test[i]);

           for (int j = 0; j < testx.length; j++) {
               int label = bayes.predict(testx[j]);
               if (label != -1) {
                   total++;
                   if (testy[j] != label) {
                       error++;
                   }
               }
           }
       }

       System.out.format("Bernoulli error = %d of %d%n", error, total);
       assertTrue(error < 270);
   }
}

