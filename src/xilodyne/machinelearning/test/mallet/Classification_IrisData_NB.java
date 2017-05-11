package xilodyne.machinelearning.test.mallet;

import java.util.Iterator;

import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
http://www.cs.umass.edu/~mccallum/mallet
This software is provided under the terms of the Common Public License,
version 1.0, as published by http://www.opensource.org.  For further
information, see the file `LICENSE' included with this distribution. */




/** 
@author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
*/

public class Classification_IrisData_NB {

	public static void main(String[] args) {
		Classification_IrisData_NB.testRandomTrained();

	}
	
	public static void testRandomTrained ()
	{
		InstanceList ilist = new InstanceList (new Randoms(1), 10, 2);
		printInstanceList(ilist);
		Classifier c = new NaiveBayesTrainer ().train (ilist);
		// test on the training data
		int numCorrect = 0;
		for (int i = 0; i < ilist.size(); i++) {
			Instance inst = ilist.get(i);
			Classification cf = c.classify (inst);
			cf.print ();
			if (cf.getLabeling().getBestLabel() == inst.getLabeling().getBestLabel())
				numCorrect++;
		}
		System.out.println ("Accuracy on training set = " + ((double)numCorrect)/ilist.size());
	}
	
	private static void printInstanceList(InstanceList iList) {
		Iterator<Instance> list = iList.iterator();
		while (list.hasNext() ) {
			Instance inst = list.next();
			System.out.println(inst.getName());
		}
		
	}

}
