package model.net.domain;

import model.net.iActivationFunction;
import model.net.iHiddenLayer;
import model.net.utils.MatrixUtils;

public class HiddenLayer extends AbstractLayer implements iHiddenLayer {

	// fields

	// ------------------------------------
	// constructor
	public HiddenLayer() {
	}

	// ------------------------------------
	// methods
	@Override
	public void setActiveFunction(iActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	@Override
	public void computeError() throws Exception {
		this.errorVector = computeErrorHelper(nextLayer.getWeightMatrix(), nextLayer.getErrorVector());
	}

	private double[] computeErrorHelper(double[][] weightMatrix, double[] errorVector) throws Exception {

		double[] result = new double[size];
		
		double[][] tranposeWeightMatrix = MatrixUtils.tranposeMatrix(weightMatrix);
		double[] multipleTemp = MatrixUtils.multiple(tranposeWeightMatrix, errorVector);
		double[] divired = activationFunction.computeDerived(outputVector);

		//
		for (int i = 0; i < size; i++) {
			result[i] = multipleTemp[i] * divired[i];
		}

		return result;
	}

}
