package model.net.domain;

import model.net.iActivationFunction;
import model.net.iLayer;
import model.net.utils.MatrixUtils;

public class AbstractLayer implements iLayer {

	// fields
	// number neural in layer
	protected int size;

	// activation function
	protected iActivationFunction activationFunction;
	
	// weight matrix - all weight which all neural in this layer connected to all
	// neural in previous layer
	protected double[][] weightMatrix;

	// delta matrix - error of all weight
	protected double[][] deltaMatrix;

	// bias vector - bias for all neural in this layer
	protected double[] biasVector;

	// delta vector - bias's error of all neural in this layer
	protected double[] deltaVector;

	// error vector - error of all neural in this layer
	protected double[] errorVector;

	// output vector - output of all neural in this layer
	protected double[] outputVector;

	// next layer
	protected iLayer nextLayer;

	// previous layer
	protected iLayer previousLayer;

	// -----------------------------------------
	// constructor
	protected AbstractLayer() {
		this.size = 0;
		this.activationFunction = SigmoidFunction.getInstance();
		this.weightMatrix = null;
		this.deltaMatrix = null;
		this.biasVector = null;
		this.deltaVector = null;
		this.errorVector = null;
		this.outputVector = null;
		this.nextLayer = null;
	}

	// -----------------------------------------
	// methods
	// reset layer (create weight matrix and create delta matrix)
	@Override
	public void createLayer() {
		int previousLayerSize = this.previousLayer.getSize();

		// set up weight matrix
		this.weightMatrix = new double[this.size][previousLayerSize];
		for (int i = 0; i < weightMatrix.length; i++) {
			for (int j = 0; j < weightMatrix[0].length; j++) {
				weightMatrix[i][j] = -0.5 + Math.random();
			}
		}

		// set up bias vector
		this.biasVector = new double[this.size];
		for (int i = 0; i < biasVector.length; i++) {
			biasVector[i] = -0.5 + Math.random();
		}

		// set up delta matrix
		this.deltaMatrix = new double[this.size][previousLayerSize];

		// set up delta vector
		this.deltaVector = new double[this.size];
	}

	// set size
	@Override
	public void setSize(int size) {
		this.size = size;
	}

	// get output
	@Override
	public double[] getOutputVector() {
		return this.outputVector;
	}

	public void computeOutput() throws Exception {
		double[] multipleVector = MatrixUtils.multiple(weightMatrix, previousLayer.getOutputVector());
		double[] plusVector = MatrixUtils.plus(multipleVector, biasVector);
		this.outputVector = activationFunction.active(plusVector);
	}

	// set next layer
	@Override
	public void setNextLayer(iLayer nextLayer) {
		this.nextLayer = nextLayer;
		nextLayer.setPreviousLayer(this);
	}

	@Override
	public void setPreviousLayer(iLayer previousLayer) {
		this.previousLayer = previousLayer;
	}

	// update methods (learning)
	@Override
	public void updateDelta() throws Exception {
		// update delta matrix
		double[][] newDeltaMatrix = computeDeltaMatrix(this.errorVector,
				this.previousLayer.getOutputVector());

		for (int i = 0; i < this.deltaMatrix.length; i++) {
			for (int j = 0; j < this.deltaMatrix[0].length; j++) {
				this.deltaMatrix[i][j] = this.deltaMatrix[i][j] + newDeltaMatrix[i][j];
			}
		}

		// update delta vector
		for (int i = 0; i < this.deltaVector.length; i++) {
			this.deltaVector[i] = this.deltaVector[i] + this.errorVector[i];
		}
	}

	private double[][] computeDeltaMatrix(double[] errorVector, double[] outputVector) {
		int numRow = errorVector.length;
		int numCol = outputVector.length;

		double[][] result = new double[numRow][numCol];

		for (int i = 0; i < numRow; i++) {
			for (int j = 0; j < numCol; j++) {
				result[i][j] = errorVector[i] * outputVector[j];
			}
		}

		return result;
	}

	@Override
	public void updateWeightAndBias(double alpha, int len) {
		// update weight
		for (int i = 0; i < this.weightMatrix.length; i++) {
			for (int j = 0; j < this.weightMatrix[0].length; j++) {
				this.weightMatrix[i][j] = this.weightMatrix[i][j] - (alpha / len) * this.deltaMatrix[i][j];
			}
		}

		// update bias
		for (int i = 0; i < this.biasVector.length; i++) {
			this.biasVector[i] = this.biasVector[i] - (alpha / len) * this.deltaVector[i];
		}

		// reset deltaMatrix
		this.deltaMatrix = new double[this.size][this.previousLayer.getSize()];

		// reset deltaVector
		this.deltaVector = new double[this.size];
	}

	// get methods
	public int getSize() {
		return size;
	}

	public double[][] getWeightMatrix() {
		return weightMatrix;
	}

	public double[][] getDeltaMatrix() {
		return this.deltaMatrix;
	}

	public double[] getBiasVector() {
		return biasVector;
	}

	public double[] getDeltaVector() {
		return deltaVector;
	}

	public double[] getErrorVector() {
		return this.errorVector;
	}

	public iLayer getNextLayer() {
		return nextLayer;
	}

	public iLayer getPreviousLayer() {
		return previousLayer;
	}
}
