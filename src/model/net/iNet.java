package model.net;

public interface iNet {

	// get information about net
	public String getInfo();

	// create a neural net
	public void setActiveFunction(iActivationFunction activationFunction);

	public void setNumberInput(int number);
	
	public void addHiddenLayer(int numberNeural);

	public void setNumberOutput(int number);
	
	public void createNet();

	// working methods
	public double[] predict(double[] data) throws Exception;

	public void train(double[][] dataSet, double[][] labelSet, int epoch, int miniBatch, double alpha) throws Exception;

	public void saveCurrentNet(String currentNetPath) throws Exception;

	public void loadTrainedNet(String trainedNetPath) throws Exception;

	public int test(double[][] dataTestSet, double[] labelTestSet) throws Exception;
}
