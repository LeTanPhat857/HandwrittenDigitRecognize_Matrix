package model.net.utils;

import java.io.IOException;

public class MatrixUtils {

	public static double[] plus(double[] vector1, double[] vector2) throws IOException {
		int size1 = vector1.length;
		int size2 = vector2.length;

		if (size1 != size2) {
			throw new IOException("Vector 1 has size: " + size1 + "; Vector 2 has size: " + size2);
		}

		double[] result = new double[size1];

		for (int i = 0; i < result.length; i++) {
			result[i] = vector1[i] + vector2[i];
		}

		return result;
	}

	public static double[] multiple(double[][] matrix, double[] vector) throws Exception {
		int length = matrix.length;
		double[] result = new double[length];

		for (int row = 0; row < length; row++) {
			double[] weightVector = matrix[row];
			result[row] = multiple(weightVector, vector);
		}

		return result;
	}

	public static double multiple(double[] vector1, double[] vector2) throws Exception {
		int size1 = vector1.length;
		int size2 = vector2.length;

		if (size1 != size2) {
			throw new IOException("Vector 1 has size: " + size1 + "; Vector 2 has size: " + size2);
		}

		double result = 0.0;

		for (int i = 0; i < size1; i++) {
			result += vector1[i] * vector2[i];
		}

		return result;
	}

//	public static double computeCost(double[] predictVec, double[] desiredVec) {
//		double result = 0;
//
//		for (int i = 0; i < predictVec.length; i++) {
//			result += -desiredVec[i] * Math.log10(predictVec[i])
//					- (1 - desiredVec[i]) * (Math.log10((1 - predictVec[i])));
//		}
//
//		return result;
//	}

	public static double[][] tranposeMatrix(double[][] matrix) {
		int numRow = matrix[0].length;
		int numCol = matrix.length;

		double[][] result = new double[numRow][numCol];

		for (int i = 0; i < numRow; i++) {
			for (int j = 0; j < numCol; j++) {
				result[i][j] = matrix[j][i];
			}
		}

		return result;
	}

	// test
	public static void main(String[] args) throws Exception {
		double[][] matrix = { { 1, 2, 3, 1 }, { 4, 5, 6, 1 } };
		double[] vector = { 1, 1, 3 };
		double[] vector1 = { 2, 4, 5, 6 };

//		System.out.println(Arrays.toString(multiple(tranposeMatrix(matrix), vector)));

	}

}
