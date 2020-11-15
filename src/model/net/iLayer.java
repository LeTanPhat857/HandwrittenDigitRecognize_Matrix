package model.net;

public interface iLayer {

	//
	public void setSize(int size);

	public void setNextLayer(iLayer nextLayer);

	public void setPreviousLayer(iLayer previousLayer);

	public void createLayer();

	//
	public void computeOutput() throws Exception;

	public double[] getOutputVector();

	public int getSize();

	public double[][] getWeightMatrix();

	public double[][] getDeltaMatrix();

	public double[] getBiasVector();

	public double[] getDeltaVector();

	public double[] getErrorVector();

	public iLayer getNextLayer();

	public iLayer getPreviousLayer();

	//
	public void updateDelta() throws Exception;

	public void updateWeightAndBias(double alpha, int len);
}
