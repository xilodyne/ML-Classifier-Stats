package xilodyne.machinelearning.test.corenlp;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import edu.stanford.nlp.classify.Classifier;
import edu.stanford.nlp.classify.ColumnDataClassifier;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.util.ErasureUtils;

public class ClassifierDemo {
	 public static void main(String[] args) {
		    ColumnDataClassifier cdc = new ColumnDataClassifier("data/stanford/examples/cheese2007.prop");
		    Classifier<String,String> cl =
		        cdc.makeClassifier(cdc.readTrainingExamples("data/stanford/examples/cheeseDisease.train"));
		    for (String line : ObjectBank.getLineIterator("data/stanford/examples/cheeseDisease.test")) {
		      Datum<String,String> d = cdc.makeDatumFromLine(line, 0);
		      System.out.println(line + "  ==>  " + cl.classOf(d));
		    }
		    
		    try {
				demonstrateSerialization(args);
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		  }


		  public static void demonstrateSerialization(String[] args) 
		    throws IOException, ClassNotFoundException
		  {
		    ColumnDataClassifier cdc = new ColumnDataClassifier("data/stanford/examples/cheese2007.prop");
		    Classifier<String,String> cl =
		        cdc.makeClassifier(cdc.readTrainingExamples("data/stanford/examples/cheeseDisease.train"));

		    // Exhibit serialization and deserialization working. Serialized to bytes in memory for simplicity
		    System.out.println(); System.out.println();
		    ByteArrayOutputStream baos = new ByteArrayOutputStream();
		    ObjectOutputStream oos = new ObjectOutputStream(baos);
		    oos.writeObject(cl);
		    oos.close();
		    byte[] object = baos.toByteArray();
		    ByteArrayInputStream bais = new ByteArrayInputStream(object);
		    ObjectInputStream ois = new ObjectInputStream(bais);
		    LinearClassifier<String,String> lc = ErasureUtils.uncheckedCast(ois.readObject());
		    ois.close();
		    ColumnDataClassifier cdc2 = new ColumnDataClassifier("data/stanford/examples/cheese2007.prop");

		    for (String line : ObjectBank.getLineIterator("data/stanford/examples/cheeseDisease.test")) {
		      Datum<String,String> d = cdc.makeDatumFromLine(line, 0);
		      Datum<String,String> d2 = cdc2.makeDatumFromLine(line, 0);
		      System.out.println(line + "  =origi=>  " + cl.classOf(d));
		      System.out.println(line + "  =deser=>  " + lc.classOf(d2));
		    }
		  }
}
