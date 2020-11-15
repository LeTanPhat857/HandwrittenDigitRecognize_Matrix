package model.net.domain;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import model.net.iActivationFunction;
import model.net.iHiddenLayer;
import model.net.iInputLayer;
import model.net.iLayer;
import model.net.iNet;
import model.net.iOutputLayer;

public class Net implements iNet {

	// fields
	private iInputLayer inputLayer;
	private List<iHiddenLayer> hiddenLayers;
	private iOutputLayer outputLayer;

	// constructor
	public Net() {
		this.inputLayer = new InputLayer();
		this.hiddenLayers = new ArrayList<iHiddenLayer>();
		this.outputLayer = new OutputLayer();
	}

	// methods
	@Override
	public String getInfo() {
		// TODO Auto-generated method stub
		return null;
	}

	// set up net
	@Override
	public void setActiveFunction(iActivationFunction activationFunction) {
		for (iHiddenLayer hiddenLayer : hiddenLayers) {
			hiddenLayer.setActiveFunction(activationFunction);
		}
	}

	@Override
	public void setNumberInput(int number) {
		this.inputLayer.setSize(number);
	}

	@Override
	public void addHiddenLayer(int number) {
		iHiddenLayer hiddenLayer = new HiddenLayer();
		hiddenLayer.setSize(number);
		this.hiddenLayers.add(hiddenLayer);
	}

	@Override
	public void setNumberOutput(int number) {
		this.outputLayer.setSize(number);
	}

	@Override
	public void createNet() {
		if (this.hiddenLayers.isEmpty()) { // if not hidden layer
			this.inputLayer.setNextLayer(this.outputLayer);
			this.outputLayer.createLayer();
		} else { // if have hidden layers
			int lenHiddenList = this.hiddenLayers.size();

			// input layer connected to first hidden layer
			this.inputLayer.setNextLayer(this.hiddenLayers.get(0));
			this.hiddenLayers.get(0).createLayer();

			// hidden layer connected to next hidden layer
			for (int i = 0; i < lenHiddenList - 1; i++) {
				this.hiddenLayers.get(i).setNextLayer(this.hiddenLayers.get(i + 1));
				this.hiddenLayers.get(i + 1).createLayer();
			}

			// last hidden layer connected to output layer
			this.hiddenLayers.get(lenHiddenList - 1).setNextLayer(outputLayer);
			this.outputLayer.createLayer();
		}
	}

	// working methods
	@Override
	public double[] predict(double[] data) throws Exception {
		// set data for input layer
		this.inputLayer.setOutputVector(data);

		// compute output for all hidden layer
		for (int i = 0; i < this.hiddenLayers.size(); i++) {
			this.hiddenLayers.get(i).computeOutput();
		}

		// compute output for output layer
		this.outputLayer.computeOutput();

		// return final result
		return this.outputLayer.getOutputVector();
	}

	@Override
	public void train(double[][] dataSet, double[][] labelSet, int epoch, int miniBatch, double alpha)
			throws Exception {

		// length of dataSet and labelSet
		int len = dataSet.length;

		// for 1 epoch
		for (int i = 0; i < epoch; i++) {
//			System.out.println("epoch: " + i);

			// shuffle data
			shuffleTrainingData(dataSet, labelSet);

			// for 1 miniBatch in (lenOfData / miniBatch)
			for (int j = 0; j < len; j += miniBatch) {
				// creating miniBatch
				double[][] dataBatch = Arrays.copyOfRange(dataSet, j, j + miniBatch);
				double[][] labelBatch = Arrays.copyOfRange(labelSet, j, j + miniBatch);

				// training miniBatch
				trainingMiniBatch(dataBatch, labelBatch, alpha);
			}
		}
	}

	private void shuffleTrainingData(double[][] dataSet, double[][] labelSet) {
		int len = dataSet.length;

		for (int i = 0; i < len; i++) {
			int randNum = new Random().nextInt(len);

			double[] tempData = dataSet[i];
			dataSet[i] = dataSet[randNum];
			dataSet[randNum] = tempData;

			double[] tempLabel = labelSet[i];
			labelSet[i] = labelSet[randNum];
			labelSet[randNum] = tempLabel;
		}
	}

	private void trainingMiniBatch(double[][] dataBatch, double[][] labelBatch, double alpha) throws Exception {
		// length of miniBatch
		int len = dataBatch.length;

		// for 1 data and 1 label in miniBatch
		for (int i = 0; i < len; i++) {
			// step 1. forward propagation
			double[] predictVec = predict(dataBatch[i]);
			double[] desiredVec = labelBatch[i];

			// step 2. back propagation
			// step 2.1. compute error and update delta for outputLayer
			this.outputLayer.computeError(predictVec, desiredVec);
			this.outputLayer.updateDelta();

			// step 2.2. compute error and update delta for all hidden layer
			for (int k = hiddenLayers.size() - 1; k >= 0; k--) {
				hiddenLayers.get(k).computeError();
				hiddenLayers.get(k).updateDelta();
			}
		}

		// step 3. gradient descent
		// step 3.1. gradient descent for outputLayer
		this.outputLayer.updateWeightAndBias(alpha, len);

		// step 3.2 gradient descent for hiddenLayers
		for (int k = hiddenLayers.size() - 1; k >= 0; k--) {
			hiddenLayers.get(k).updateWeightAndBias(alpha, len);
		}
	}

	@Override
	public void saveCurrentNet(String trainedNet) throws IOException {
		File file = new File(trainedNet);
		System.out.println(file.createNewFile());

		FileWriter fileWriter = new FileWriter(file);

		// set up net
		fileWriter.write(this.inputLayer.getSize() + " ");

		int numberHiddenLayer = hiddenLayers.size();
		fileWriter.write(numberHiddenLayer + " ");

		for (int i = 0; i < numberHiddenLayer; i++) {
			fileWriter.write(this.hiddenLayers.get(i).getSize() + " ");
		}

		fileWriter.write(this.outputLayer.getSize() + "\n");

		// bias vector and weight matrix
		for (int i = 0; i < numberHiddenLayer + 1; i++) {
			iLayer layer = null;

			if (i < numberHiddenLayer) {
				layer = this.hiddenLayers.get(i);
			} else {
				layer = this.outputLayer;
			}

			// write bias vector
			double[] biasVector = layer.getBiasVector();
			int numberNeural = biasVector.length;

			fileWriter.write(numberNeural + "\n");
			
			for (int j = 0; j < numberNeural; j++) {
				fileWriter.write(biasVector[j] + " ");
			}
			
			fileWriter.write("\n");
			
			// write weight matrix
			double[][] weightMatrix = layer.getWeightMatrix();
			int numRow = weightMatrix.length;
			int numCol = weightMatrix[0].length;

			fileWriter.write(numRow + " " + numCol + "\n");

			for (int row = 0; row < numRow; row++) {
				for (int col = 0; col < numCol; col++) {
					fileWriter.write(weightMatrix[row][col] + " ");
				}

				fileWriter.write("\n");
			}
		}

		fileWriter.close();
	}

	@Override
	public void loadTrainedNet(String trainedNet) throws FileNotFoundException {
		File file = new File(trainedNet);
		Scanner scanner = new Scanner(file);

		// set up net
		int numberInput = scanner.nextInt();
		this.inputLayer.setSize(numberInput);

		int numberHiddenLayer = scanner.nextInt();
		for (int i = 0; i < numberHiddenLayer; i++) {
			int hiddenLayerSize = scanner.nextInt();
			this.addHiddenLayer(hiddenLayerSize);
		}

		int numberOutput = scanner.nextInt();
		this.outputLayer.setSize(numberOutput);

		this.createNet();

		// set weight matrix
		for (int i = 0; i < numberHiddenLayer + 1; i++) {
			iLayer layer = null;

			if (i < numberHiddenLayer) {
				layer = this.hiddenLayers.get(i);
			} else {
				layer = this.outputLayer;
			}

			// read bias
			double[] biasVector = layer.getBiasVector();
			
			int numberNeural = scanner.nextInt();
			
			for (int j = 0; j < numberNeural; j++) {
				biasVector[j] = scanner.nextDouble(); 
			}
			
			// read weight
			double[][] weightMatrix = layer.getWeightMatrix();

			int numRow = scanner.nextInt();
			int numCol = scanner.nextInt();

			for (int row = 0; row < numRow; row++) {
				for (int col = 0; col < numCol; col++) {
					weightMatrix[row][col] = scanner.nextDouble();
				}
			}
		}
	}

	@Override
	public int test(double[][] dataTestSet, double[] labelTestSet) throws Exception {
		int counter = 0;

		for (int i = 0; i < dataTestSet.length; i++) {

//			// find max
			double[] predict = this.predict(dataTestSet[i]);
			int maxIndex = 0;
			double maxValue = predict[0];

			for (int j = 1; j < 10; j++) {
				if (maxValue < predict[j]) {
					maxIndex = j;
					maxValue = predict[j];
				}
			}

			// compare
			if (maxIndex == (int) labelTestSet[i]) {
				counter++;
			}
//			if (predict[(int) labelTestSet[i]] > 0.8000000) {
//				counter++;
//			}
		}

		return counter;
	}

	// get methods
	public iInputLayer getInputLayer() {
		return inputLayer;
	}

	public List<iHiddenLayer> getHiddenLayers() {
		return hiddenLayers;
	}

	public iOutputLayer getOutputLayer() {
		return outputLayer;
	}
}
