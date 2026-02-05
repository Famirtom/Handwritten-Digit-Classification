import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/** 
 * 
 * Project Name: Machine Learning Project
 * 
 * This system implements multiple machine learning algorithms to classify handwritten digits  (0-9) form two CSV datasets.
 * Each dataset contains 8X8 pixel images represented as 65 numerical values (64 pixel + 1 label).
 * 
 * Implemented Algorithms
 * 1. Eucliedean Distance - Measures pixel similarity between two digits
 * 2. K-Nearest Neighbors (KNN), classifies a digit based on the majority vote of its closest neighbors.
 * 3. Genetic Algorithm, optimizes the number of nearby samples (K) using selection, crossover and mutation.
 * 4. Multi-Layer Perceptron (MLP) 
 * 
 * This program also applies two-fold cross-validation:
 * - Dataset1 is used for training and Dataset2 for testing.
 * - Then roles are swapped, and results are averaged
 * 
 */


public class CourseWorkDigits {
	
		// Dataset configuration
		private static final String DATASET1_PATH = "src/dataSet1.csv";
		private static final String DATASET2_PATH = "src/dataSet2.csv";
		private static final int TOTAL_FEATURES = 65;
		private static final int INPUT_FEATURES = TOTAL_FEATURES -1;
		private static final int NUM_CLASSES = 10;

		private static final int HIDDEN_LAYER_SIZE = 32;
		private static final double LEARNING_RATE = 0.05;
		private static final int MAX_EPOCHS = 100;
		
		// Genetic algorithm configuration
		private static final int POPULATION_SIZE = 10;
		private static final int GA_GENERATIONS = 15;
		private static final double MUTATION_RATE = 0.1;
				
		private static int[][] data;
		// Random fixed random seed to make the algorithm's behavior reproducible during testing
		private static final int RANDOM_SEED = 42;
		private static Random random = new Random(RANDOM_SEED);
		
		// Configuration to explore nearby values between 1 and 20.
		private static final int MIN_NEARBYVALUES = 1;
		private static final int MAX_NEARBYVALUES = 20;
		
		private static final int MUTATION_STEP = 2;
		
		/**
		 * Reads CSV file and converts it to a 2D integer array
		 * @param fileName
		 */
		public static void loadDataset(String fileName){
			ArrayList<int[]> dataList = new ArrayList<>();
			try {
				File fileHandle = new File(fileName);
				Scanner scanner = new Scanner(fileHandle);
				while(scanner.hasNextLine()) {
					String line = scanner.nextLine();
					String [] values = line.split(","); 
					int[] row = new int[values.length]; 

					for(int pixel =0; pixel <values.length; pixel ++) { 
						row[pixel] = Integer.parseInt(values[pixel]); 
					}
					dataList.add(row); 
				}
				 data = new int[dataList.size()][TOTAL_FEATURES]; 
				for(int sampleIndex=0; sampleIndex< dataList.size(); sampleIndex++) {
					data[sampleIndex] = dataList.get(sampleIndex);
				}
				scanner.close();
			}
			catch(FileNotFoundException error){
				System.out.println("ERROR: File Not Found! " + error);
			}
		}
		
		/**
		 * Calculates Euclidean distance between two feature vectors
		 * Formula: sqrt(sum(xi - yi)^2))
		 * @param testInstance
		 * @param trainInstance
		 * @return Euclidean Distance
		 */
		public static double euclideanDistance(int [] testInstance, int [] trainInstance) {
			double sumSquareDifference=0;
			for(int feature = 0; feature < INPUT_FEATURES ;feature++) {
				double diff = testInstance[feature] - trainInstance[feature];
				sumSquareDifference = sumSquareDifference + (diff * diff);
			}
			return Math.sqrt(sumSquareDifference);
		}
		
		/**
		 * K-Nearest Neighbors Classification Algorithm
		 * Steps:
		 * 1. Calculate distances to all training points
		 * 2. Sort the k points using Bubble Sort
		 * 3. for each remaining training points:
		 * 	- Calculate distance to test point
		 * 	- If closer than the furthest of out k neighbors, replace it
		 * 	- Re/sort to maintain order using Insertion Sort
		 * 4. Count votes form k nearest neighbors
		 * 5. predict the class with most votes
		 * @param trainData training dataset
		 * @param testData test dataset
		 * @param k Number of neighbors to consider
		 * @param showConfusionMatrix whether to display confusion matrix 
		 * @return Classification accuracy as percentage
		 */
		public static double knnClassification(int[][] trainData, int[][] testData , int nearbySamples, boolean showConfusionMatrix) {
			int totalTests = 0;
			int correctPredictions = 0;
			int [][] confusionMatrix = new int[NUM_CLASSES][NUM_CLASSES];
			for(int testIndex=0; testIndex < testData.length; testIndex++) {
				DistanceLabel [] kNearest = new DistanceLabel[nearbySamples];
				// Calculate distances to all training points
				for(int trainIndex=0; trainIndex < nearbySamples; trainIndex++) {
					double dist = euclideanDistance(testData[testIndex], trainData[trainIndex]);
					int label = trainData[trainIndex][INPUT_FEATURES];
					kNearest[trainIndex] = new DistanceLabel(dist, label);
				}	
				// Sort initial k neighbors using bubble Sort
				bubbleSort(kNearest);
				// check remaining training points
				// Only keep them if they are closer than the furthest of our k neighbors
				for(int trainIndex = nearbySamples; trainIndex < trainData.length; trainIndex ++) {
					double dist = euclideanDistance(testData[testIndex], trainData[trainIndex]);	
					//if this point is close then the furthest neighbor
					if(dist < kNearest[nearbySamples-1].distance) {
						int label = trainData[trainIndex][INPUT_FEATURES];
						kNearest[nearbySamples-1]= new DistanceLabel(dist, label);
						// Re-Sort to maintain order
						insertionSortLast(kNearest);
					}
				}
				// Count votes from k nearest neighbors
				int[] classVotes =  new int[NUM_CLASSES];
				for( int neighbor =0 ; neighbor < nearbySamples; neighbor ++) {
					int neighborLabel = kNearest[neighbor].label;
					classVotes[neighborLabel]++;
				}
				// Find class with most votes
				int predictedClass = 0;
				int maxVotes = classVotes[0];
				for(int classIndex = 0; classIndex < NUM_CLASSES; classIndex++) {
					if(classVotes[classIndex] > maxVotes) {
						maxVotes = classVotes[classIndex];
						predictedClass = classIndex;
					}
				}
				int actualClass = testData[testIndex][INPUT_FEATURES];
				confusionMatrix[actualClass][predictedClass]++;
				if(predictedClass == actualClass) {
					correctPredictions++;
				}
				totalTests++;
		}
			if(showConfusionMatrix) {
				printConfusionMatrix(confusionMatrix);
			}
			return (double) correctPredictions / totalTests * 100.0;
	}		
		
		/**
		 * Bubble Sort algorithm
		 * This algorithm will repeatedly steps through the array,
		 * compares adjacent elements and swap them if they are in wrong order.
		 * the pass through the array is repeated until the array is sorted
		 *@param array of DistanceLabel objects to sort by distance.
		 */
		private static void bubbleSort(DistanceLabel[] array) {
		    int n = array.length;
		    for(int first = 0; first < n - 1; first++) {
		        for(int second = 0; second < n - first - 1; second++) { 
		            if(array[second].distance > array[second + 1].distance) {
		                // Swap elements
		                DistanceLabel temp = array[second];
		                array[second] = array[second + 1];
		                array[second + 1] = temp;
		            }
		        }
		    }
		}
		
		/** 
		 * Insertion Sort for the last element
		 * Is efficient as the array is already sorted.
		 * Used to maintain sorted order when we replace the furthest neighbor with a close one
		 * @param array Sorted array 
		 */
		private static void insertionSortLast(DistanceLabel[] array) {
			DistanceLabel newElement = array[array.length -1];
			int index = array.length -2;
			
			// move elements greater then key one position ahead
			while(index >= 0 && array[index].distance > newElement.distance) {
				array[index +1] = array[index];
				index--;
			}
			array[index + 1] = newElement;
		}
		
		/**
		 * Print Confusion matrix showing predicted vs actual classification
		 * @param matrix
		 */
		public static void printConfusionMatrix(int[][] matrix) {
			System.out.println("Confusion Matrix (Actual - Predicted)");
			System.out.println("-----------------------------------");
			System.out.printf("%8s", " ");
			for(int feature = 0; feature < NUM_CLASSES ; feature++) {
				System.out.printf(" %6d ", feature);
			}
			System.out.println();
			for(int actual =0; actual < NUM_CLASSES; actual ++) {
				System.out.printf("Actual %d | ", actual);
				for(int predicted = 0 ; predicted < NUM_CLASSES; predicted++) {
					System.out.printf("%8d",matrix[actual][predicted]);
				}
				System.out.println();
			}
			System.out.println("---------------------------------------------------------------");
			System.out.println("Legenda: Rows = True Labels,  Columns = Predicted Labels");
			System.out.println("---------------------------------------------------------------");
		}
		
		/**
		 * Print the matrix
		 * @param data
		 * @param rows
		 * @param cols
		 */
		public static void printData(int [][] data, int rows, int cols) {
			for(int row =0; row < rows; row++) {
				for(int col =0; col < cols; col++) {
					System.out.print(data[row][col]+ " ");
				}
				System.out.println();
			  }
		}
		
		/**
		 * Simple Multi-Layer Perceptron
		 * Implements forward and backward propagation for classification
		 * Layer:
		 * -input layers: 64 neurons (pixel values)
		 * -hidden layer: computes internal feature form the input pixel, 32 neurons
		 * -output layer: predict the digit probability, 10 neurons, one for each digit
		 */
		static class MultiLayerPerceptron{
			private double[][] WEIGHTS_INPUTS_HIDDEN;
			private double[][] WEIGHTS_HIDDEN_OUTPUT;
			private double[] BIAS_HIDDEN;
			private double[] BIAS_OUTPUT;
			
			public MultiLayerPerceptron() {
				WEIGHTS_INPUTS_HIDDEN = new double[INPUT_FEATURES][HIDDEN_LAYER_SIZE];
				WEIGHTS_HIDDEN_OUTPUT = new double[HIDDEN_LAYER_SIZE][NUM_CLASSES];
				BIAS_HIDDEN = new double[HIDDEN_LAYER_SIZE];
				BIAS_OUTPUT = new double[NUM_CLASSES];
				initializeWeights();
			}
			
			/** Initialize all weights randomly between -0.5 and 0.5 */
			private void initializeWeights() {
				for(int inputIndex = 0; inputIndex < INPUT_FEATURES; inputIndex++) {
					for(int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
						WEIGHTS_INPUTS_HIDDEN[inputIndex][hiddenNeuron] = (random.nextDouble() - 0.5);
					}
				}
				for(int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
					for(int outputNeuron =0 ; outputNeuron < NUM_CLASSES; outputNeuron++) {
						WEIGHTS_HIDDEN_OUTPUT[hiddenNeuron][outputNeuron] = (random.nextDouble() - 0.5);
					}
				}
					for(int hiddenNeuron =0; hiddenNeuron< HIDDEN_LAYER_SIZE; hiddenNeuron++) {
						BIAS_HIDDEN[hiddenNeuron] = 0.0;
					}
					for(int outputNeuron=0; outputNeuron< NUM_CLASSES; outputNeuron++) {
						BIAS_OUTPUT[outputNeuron] = 0.0;
					}
				}
			
			/**
			 * Sigmoid activation function, convert every number to a value between 0 and 1
			 * Reference:  https://en.wikipedia.org/wiki/Sigmoid_function
			 */
			private double sigmoid(double x) {
				return 1.0 / (1.0 +  Math.exp(-x));
			}
			// calculate the slope of the sigmoid curve 
			private double sigmoidDerivative(double output) {
				return output * (1.0 - output);
			}
			
			/**
			 * forward propagation: computes output 
			 * @param inputs nomalised pixel values 0-1
			 * @return output probabilities for each digits classe 0-9
			 */
			private double[] forward(double[] inputs) {
				double[] HIDDEN_OUTPUTS = new double[HIDDEN_LAYER_SIZE];
				double[] FINAL_OUTPUTS = new double[NUM_CLASSES];
				
				// compute activations for hidden layer
				for(int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE ; hiddenNeuron++) {
					double sum = BIAS_HIDDEN[hiddenNeuron];
					for(int inputNeuron = 0 ; inputNeuron < INPUT_FEATURES; inputNeuron++) {
						sum = sum + inputs[inputNeuron] * WEIGHTS_INPUTS_HIDDEN[inputNeuron][hiddenNeuron];
					}
					HIDDEN_OUTPUTS[hiddenNeuron] = sigmoid(sum);
				}
				
				//output layer
				for(int outputNeuron= 0 ; outputNeuron < NUM_CLASSES; outputNeuron++) {
					double sum = BIAS_OUTPUT[outputNeuron];
					for(int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
						sum = sum + HIDDEN_OUTPUTS[hiddenNeuron] * WEIGHTS_HIDDEN_OUTPUT[hiddenNeuron][outputNeuron];
					}
					FINAL_OUTPUTS[outputNeuron] = sigmoid(sum);
				}
				return FINAL_OUTPUTS;	
			}
			
			/**
			 * train the network using backpropagation
			 * @param trainData 
			 * @param epochs number of learning iterations
			 * @param learningrate speed of learning
			 */
			public void train(int[][] trainData, int epochs, double learningRate) {
				for(int epoch=0; epoch < epochs; epoch++) {
					double TOTAL_ERROR = 0.0;
					
					for(int sampleIndex = 0; sampleIndex < trainData.length; sampleIndex++) {
						int[] currentSample = trainData[sampleIndex];

			            // Normalize inputs between 0–1
			            double[] inputs = new double[INPUT_FEATURES];
			            for (int featureIndex = 0; featureIndex < INPUT_FEATURES; featureIndex++) {
			                inputs[featureIndex] = currentSample[featureIndex] / 16.0;
			            }
			            int trueLabel = currentSample[INPUT_FEATURES];

			            // Forward pass
			            double[] hiddenOutputs = new double[HIDDEN_LAYER_SIZE];
			            double[] finalOutputs = new double[NUM_CLASSES];

			            for (int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
			                double sum = BIAS_HIDDEN[hiddenNeuron];
			                for (int inputNeuron = 0; inputNeuron < INPUT_FEATURES; inputNeuron++) {
			                    sum += inputs[inputNeuron] * WEIGHTS_INPUTS_HIDDEN[inputNeuron][hiddenNeuron];
			                }
			                hiddenOutputs[hiddenNeuron] = sigmoid(sum);
			            }

			            for (int outputNeuron = 0; outputNeuron < NUM_CLASSES; outputNeuron++) {
			                double sum = BIAS_OUTPUT[outputNeuron];
			                for (int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
			                    sum += hiddenOutputs[hiddenNeuron] * WEIGHTS_HIDDEN_OUTPUT[hiddenNeuron][outputNeuron];
			                }
			                finalOutputs[outputNeuron] = sigmoid(sum);
			            }
			            // target output
			            double[] TARGET_OUTPUTS = new double[NUM_CLASSES];
			            TARGET_OUTPUTS[trueLabel]= 1.0;
			            
			            // output layer error
			            double[] OUTPUT_ERRORS = new double[NUM_CLASSES];
			            for (int outputNeuron = 0; outputNeuron < NUM_CLASSES; outputNeuron++) {
			                double error = TARGET_OUTPUTS[outputNeuron] - finalOutputs[outputNeuron];
			                OUTPUT_ERRORS[outputNeuron] = error * sigmoidDerivative(finalOutputs[outputNeuron]);
			                TOTAL_ERROR = TOTAL_ERROR + error * error;
			            }
			            
			            // Hidden layer error
			            double[] HIDDEN_ERRORS = new double[HIDDEN_LAYER_SIZE];
			            for (int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
			                double errorSum = 0.0;
			                for (int outputNeuron = 0; outputNeuron < NUM_CLASSES; outputNeuron++) {
			                    errorSum += OUTPUT_ERRORS[outputNeuron] * WEIGHTS_HIDDEN_OUTPUT[hiddenNeuron][outputNeuron];
			                }
			                HIDDEN_ERRORS[hiddenNeuron] = errorSum * sigmoidDerivative(hiddenOutputs[hiddenNeuron]);
			            }
			            
			            // Update weights (Hidden → Output)
			            for (int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
			                for (int outputNeuron = 0; outputNeuron < NUM_CLASSES; outputNeuron++) {
			                    WEIGHTS_HIDDEN_OUTPUT[hiddenNeuron][outputNeuron] += 
			                        learningRate * OUTPUT_ERRORS[outputNeuron] * hiddenOutputs[hiddenNeuron];
			                }
			            }
			            
			         // Update weights (Input → Hidden)
			            for (int inputNeuron = 0; inputNeuron < INPUT_FEATURES; inputNeuron++) {
			                for (int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
			                    WEIGHTS_INPUTS_HIDDEN[inputNeuron][hiddenNeuron] += 
			                        learningRate * HIDDEN_ERRORS[hiddenNeuron] * inputs[inputNeuron];
			                }
			            }
			         // Update biases
			            for (int outputNeuron = 0; outputNeuron < NUM_CLASSES; outputNeuron++) {
			                BIAS_OUTPUT[outputNeuron] += learningRate * OUTPUT_ERRORS[outputNeuron];
			            }
			            for (int hiddenNeuron = 0; hiddenNeuron < HIDDEN_LAYER_SIZE; hiddenNeuron++) {
			                BIAS_HIDDEN[hiddenNeuron] += learningRate * HIDDEN_ERRORS[hiddenNeuron];
			            }
			        }
					
					if (epoch % 10 == 0) {
			            System.out.printf("Epoch %d | Total Error: %.4f%n", epoch, TOTAL_ERROR);
			        }
			}
		}
			
			/** Predicts the digit class for a new input */
			public int predict(int[] sample) {
			    double[] inputs = new double[INPUT_FEATURES];
			    for (int featureIndex = 0; featureIndex < INPUT_FEATURES; featureIndex++) {
			        inputs[featureIndex] = sample[featureIndex] / 16.0;
			    }

			    double[] outputs = forward(inputs);

			    int predictedClass = 0;
			    double maxActivation = outputs[0];
			    for (int outputNeuron = 1; outputNeuron < NUM_CLASSES; outputNeuron++) {
			        if (outputs[outputNeuron] > maxActivation) {
			            maxActivation = outputs[outputNeuron];
			            predictedClass = outputNeuron;
			        }
			    }
			    return predictedClass;
			}
		

			/** Evaluates model accuracy on test data */
			public double evaluate(int[][] testData) {
			    int correctPredictions = 0;
			    for (int sampleIndex = 0; sampleIndex < testData.length; sampleIndex++) {
			        int[] sample = testData[sampleIndex];
			        int predicted = predict(sample);
			        int actual = sample[INPUT_FEATURES];
			        if (predicted == actual) {
			            correctPredictions++;
			        }
			    }
			    return (double) correctPredictions / testData.length * 100.0;
			}
		}
		
		/**
	     * Genetic Algorithm for optimizing K value in KNN
	     * this algorithm evolves a population of nearby samples (k) values over multiple generations to find the optimal one
	     * Individual: a single nearby value (integer)
	     * Population: collection of nearby values
	     * Fitness: how good a nearby sample value is measured by KNN accuracy
	     * Generation: One iteration of evolution using mutation, crossover and selection.
	     * Mutation: Random 
	     */
	    static class GeneticAlgorithm {
	        private int[] population; // array of nearby values individuals
	        private double[] fitness; // array of accuracy score
	        public GeneticAlgorithm() {
	            population = new int[POPULATION_SIZE];
	            fitness = new double[POPULATION_SIZE];
	            initializePopulation();
	        }
	        
	        // create a initial population with random nearby values
	        private void initializePopulation() {
	            for(int individualIndex = 0; individualIndex < POPULATION_SIZE; individualIndex++) {
	            	// each individual gets a random nearby value between 1 and 20
	                population[individualIndex] = random.nextInt(20) + 1; // K between 1 and 20
	            }
	        }
	        /**
	         * Evaluate how good each nearby value is by running KNN
	         * @param trainData // training dataset for KNN
	         * @param testData // test dataset for KNN
	         */
	        private void evaluateFitness(int[][] trainData, int[][] testData) {
	            for(int individualIndex = 0; individualIndex < POPULATION_SIZE; individualIndex++) {
	                int currentNearbySample = population[individualIndex];
	                // Fitness = accuracy of KNN with this K value
	                fitness[individualIndex] = knnClassification(trainData, testData, currentNearbySample, false);
	            }
	        }
	        
	        /**
	         * Pick 3 random individual and returns the best one.
	         * This simulates survival of the better individuals
	         * Example:
	         * Pick nearbySamples=5 (98% accuracy), nearbySamples=12 (89% accuracy)
	         * Return nearbySamples=5 because it has highest fitness
	         * @return Index of the selected individual 
	         */
	        private int tournamentSelection() {
	            int tournamentSize = 3;
	            // start with a random individual
	            int bestIndex = random.nextInt(POPULATION_SIZE);
	            double bestFitness = fitness[bestIndex];
	            for(int competitorNum= 1; competitorNum < tournamentSize; competitorNum++) {
	                int candidateIndex = random.nextInt(POPULATION_SIZE);
	                double candidateFitness = fitness[candidateIndex];
	                // If the candidate is better , it becomes the new best
	                if(candidateFitness > bestFitness) {
	                    bestIndex = candidateIndex;
	                    bestFitness = candidateFitness;
	                }
	            }
	            return bestIndex;
	        }
	        
	        /**
	         * Crossover
	         * Combines two parent nearby sample values to create offspring nearby sample value.
	         * I randomly pick one parent's sample.
	         * @param parent1K first parent's sample value
	         * @param parent2K second parent's sample value
	         * @return offspring's nearby value
	         */
	        private int crossover(int parent1NearbySample, int parent2NearbySample) {
	        	// randomly choose one parent
	        	if (random.nextBoolean()) {
	        	    return parent1NearbySample;
	        	} else {
	        	    return parent2NearbySample;
	        	}
	        }
	        
	        /**
	         * Mutation 
	         * with 10% probability, it will randomly adjust nearby samples by -2 to +2
	         * Example:
	         * K=5 --> mutate --> k=7 (+2)
	         * k=12 --> mutate ---> k=11 (-1)
	         * @param kValue to potentially mutate
	         * @return mutated or unchanged k value
	         */
	        private int mutate(int nearbyValue) {
	        	// 10% chance of mutation
	            if(random.nextDouble() < MUTATION_RATE) {
	            	// add random value: -2, -1, 0, 1, 2
	            	int mutationChange = random.nextInt(MUTATION_STEP * 2 + 1) -  MUTATION_STEP;
	                nearbyValue = nearbyValue + mutationChange;
	                
	                // keep nearby in valid range [1, 20]
	                if (nearbyValue < MIN_NEARBYVALUES) {
	                    nearbyValue = MIN_NEARBYVALUES;
	                }
	                if (nearbyValue > MAX_NEARBYVALUES) {
	                    nearbyValue = MAX_NEARBYVALUES;
	                }
	            }
	            return nearbyValue;
	        }
	        
	        /**
	         * Optimization loop, evaluation over many generations
	         * This will evaluate fitness over all nearby value
	         * Select the best individuals via tournament
	         * Combine parents to create offspring
	         * Random change to offspring
	         * Replace old population with new generation
	         * After all generation, return the best k value found
	         * @param trainData 
	         * @param testData
	         * @return optimal k value found by evolution
	         */
	        public int optimize(int[][] trainData, int[][] testData) {
	        	// Evolution run for multiple generations
	            for(int generation = 0; generation < GA_GENERATIONS; generation++) {
	            	// How good each k value is
	                evaluateFitness(trainData, testData);
	                
	                // Create next generation through selection, crossover and mutation
	                int[] newPopulation = new int[POPULATION_SIZE];
	                
	                for(int childIndex = 0; childIndex < POPULATION_SIZE; childIndex++) {
	                	// Pick 2 good parents via tournament
	                    int parent1Index = tournamentSelection();
	                    int parent2Index = tournamentSelection();
	                    
	                    // combine parents to create child
	                    int parent1NearbySample =population[parent1Index];
	                    int parent2NearbySample = population[parent2Index];
	                    int childk = crossover(parent1NearbySample , parent2NearbySample );
	                    
	                    // Randomly modify child
	                    childk = mutate(childk);
	                    // add child to new generation
	                    newPopulation[childIndex] = childk;
	                }
	                // Replace old population with new generation
	                population = newPopulation;
	            }
	            // Return the best k value
	            evaluateFitness(trainData, testData);
	            int bestIndividualIndex = 0;
	            double bestFitnessFound = fitness[0];
	            for(int individualIndex = 1; individualIndex < POPULATION_SIZE; individualIndex++) {
	                if(fitness[individualIndex] > bestFitnessFound){
	                	bestIndividualIndex = individualIndex;
	                	bestFitnessFound = fitness[individualIndex];
	                }
	            }
	            return population[bestIndividualIndex];
	        }
	    }
	
		
		public static void main(String [] args) {
			
			System.out.println("=================================================================");
			System.out.println("CST 3170 Machine Learning Coursework - MNIST Digit Classification");
			System.out.println("=================================================================");
			
			// Load datasets
			System.out.println("loading datasets ");
			loadDataset(DATASET1_PATH);
			int[][] dataset1 = data;
			loadDataset(DATASET2_PATH);
			int[][] dataset2 = data;
			System.out.println("Dataset1: " +  dataset1.length);
			System.out.println("Dataset2: "+ dataset2.length);
			
			// KNN WITH HYPERPARAMETER
			System.out.println("=================================================================");
			System.out.println("K-Nearest Neighbors (KNN)");
			System.out.println("Using bubble sort and Insertion Sort");
			System.out.println("=================================================================");
			System.out.println("Hyperparameter Analysis (K values):");
			System.out.println("-----------------------------------");
			System.out.println(" K |  Fold1   |  Fold2   |  Avg Acc ");
			System.out.println("-----------------------------------");

			int bestNearbySample = 1;
			double bestAccuracy = 0.0;
			
			for(int nearbySamples=1; nearbySamples<= 10 ; nearbySamples++) {
				double acc1 = knnClassification(dataset1, dataset2, nearbySamples, false);
				double acc2 = knnClassification(dataset2, dataset1, nearbySamples, false);
				double avgAccuracy = (acc1 + acc2) / 2.0;
				
				System.out.printf("%2d | %.2f%% | %.2f%% | %.2f%%\n", 
                        nearbySamples, acc1, acc2, avgAccuracy);
				System.out.println("-----------------------------------");
				
				if(avgAccuracy > bestAccuracy	) {
					bestAccuracy = avgAccuracy;
					bestNearbySample = nearbySamples;
				}
			}
			
			System.out.println("Best K value: " + bestNearbySample + " with accuracy: " + String.format("%.2f%%", bestAccuracy));
			System.out.println("-----------------------------------");
	        System.out.println("2-Fold Cross Validation (K=" + bestNearbySample + ")");
	        System.out.println("-----------------------------------");
	        double knnAcc1 = knnClassification(dataset1, dataset2, bestNearbySample, true);
	        double knnAcc2 = knnClassification(dataset2, dataset1, bestNearbySample, false);
	        double knnAvg = (knnAcc1 + knnAcc2) / 2.0;
	        System.out.printf("Fold 1 Accuracy: %.2f%%", knnAcc1);
	        System.out.println();
	        System.out.printf("Fold 2 Accuracy: %.2f%%", knnAcc2);
	        System.out.println();
	        System.out.printf("Average Accuracy: %.2f%%", knnAvg);
	        System.out.println();
	        
	        // Genetic Algorithm
	        System.out.println("=================================================================");
	        System.out.println("Genetic Algorithm ");
	        System.out.println("=================================================================");
			GeneticAlgorithm ga = new GeneticAlgorithm();
			int optimizedK = ga.optimize(dataset1, dataset2);
			double gaAcc1 = knnClassification(dataset1, dataset2, optimizedK, false);
	        double gaAcc2 = knnClassification(dataset2, dataset1, optimizedK, false);
	        double gaAvg = (gaAcc1 + gaAcc2) / 2.0;
	        
	        System.out.println("GA Optimized K: " + optimizedK);
	        System.out.printf("Fold 1 Accuracy: %.2f%%\n", gaAcc1);
	        System.out.printf("Fold 2 Accuracy: %.2f%%\n", gaAcc2);
	        System.out.printf("Average Accuracy: %.2f%%\n", gaAvg);
	        
	        // MULTI-LAYER PERCEPTRON 
	        System.out.println("\n=============================================================");
	        System.out.println("MULTI-LAYER PERCEPTRON (MLP)");
	        System.out.println("=============================================================\n");
	        
	        System.out.println("Training MLP (Fold 1: train on dataset1, test on dataset2)");
	        MultiLayerPerceptron mlp1 = new MultiLayerPerceptron();
	        mlp1.train(dataset1, MAX_EPOCHS, LEARNING_RATE);
	        double mlpAcc1 = mlp1.evaluate(dataset2);
	        
	        System.out.println("\nTraining MLP (Fold 2: train on dataset2, test on dataset1)");
	        MultiLayerPerceptron mlp2 = new MultiLayerPerceptron();
	        mlp2.train(dataset2, MAX_EPOCHS, LEARNING_RATE);
	        double mlpAcc2 = mlp2.evaluate(dataset1);
	        
	        double mlpAvg = (mlpAcc1 + mlpAcc2) / 2.0;
	        System.out.printf("\nFold 1 Accuracy: %.2f%%\n", mlpAcc1);
	        System.out.printf("Fold 2 Accuracy: %.2f%%\n", mlpAcc2);
	        System.out.printf("Average Accuracy: %.2f%%\n", mlpAvg);
	        
	     // FINAL SUMMARY 
	        System.out.println("\n=============================================================");
	        System.out.println("FINAL RESULTS SUMMARY");
	        System.out.println("=============================================================\n");
	        System.out.printf("KNN (K=%d):           %.2f%%\n", bestNearbySample, knnAvg);
	        System.out.printf("GA-Optimized KNN (K=%d): %.2f%%\n", optimizedK, gaAvg);
	        System.out.printf("MLP Neural Network:   %.2f%%\n", mlpAvg);
	        System.out.println("\n=============================================================");    
		}
		
		/**
		 * Class to store distance and corresponding class label
		 */
		static class DistanceLabel{
			double distance;
			int label;
			
			DistanceLabel(double distance, int label){
				this.distance = distance;
				this.label = label;
			}
		}
}
