using Microsoft.ML;
using Shared;
using System;

namespace ImageTrainConsole
{
    class Program
    {
		// update this with your file's path where you saved it
		private static string TRAIN_DATA_FILEPATH = @"C:\Readify\mlnet-workshop\data\true_car_listings.csv";
		private static string MODEL_SAVE_FILEPATH = @"C:\Readify\mlnet-workshop\MLModel.zip";

		static void Main(string[] args)
        {
			/*
			3 Min Challenge
			Console.WriteLine("Hello World!");

			// Add input data
			var input = new ModelInput();

			// Load model and predict output of sample data
			ModelOutput result = ConsumeModel.Predict(input);
			*/

			var mlContext = new MLContext();

			// Load training data
			Console.WriteLine("Loading data...");
			IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(path: TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: ',');

			// Split the data into a train and test set. This is currently assigning 20% of the data to the training set
			var trainTestSplit = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);

			// Create data transformation pipeline
			var dataProcessPipeline =
				// Encode "Make" and "Model" columns using OneHotEncoding
				mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
					.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
					// Concat "Year", "Mileage", "MakeEncoded" and "ModelEncoded" into a column called "Features"
					.Append(mlContext.Transforms.Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
					// Normalise the "Features" column with a min value of 0 and max value of 1
					.Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
					// Cache the result (ML.NET does not cache by default)
					.AppendCacheCheckpoint(mlContext);

			/* IMPORTANT!! All input columns (features) need to be numbers, since ML.NET does not recognise text
				- This is what OneHotEncoding does (use this on category columns - i.e. columns you would use to group by)
				- For other text fields, use Featurize text
			  This is how you would handle a text column that is not a category:
				- .Append(mlContext.Transforms.Text.ApplyWordEmbedding(outputColumnName: "ModelEncoded2", inputColumnName: "Model"))
			*/

			// Choose an algorithm and add to the pipeline
			var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression();
			var trainingPipeline = dataProcessPipeline.Append(trainer);

			// Train the model - nothing is actually happening with the code above, until Fit() is called
			Console.WriteLine("Training model...");
			var model = trainingPipeline.Fit(trainTestSplit.TrainSet);

			// Make predictions on train and test sets
			IDataView trainSetPredictions = model.Transform(trainTestSplit.TrainSet);
			IDataView testSetpredictions = model.Transform(trainTestSplit.TestSet);

			// Calculate evaluation metrics for train and test sets
			var trainSetMetrics = mlContext.Regression.Evaluate(trainSetPredictions, labelColumnName: "Label", scoreColumnName: "Score");
			var testSetMetrics = mlContext.Regression.Evaluate(testSetpredictions, labelColumnName: "Label", scoreColumnName: "Score");

			Console.WriteLine($"Train Set R-Squared: {trainSetMetrics.RSquared} | Test Set R-Squared {testSetMetrics.RSquared}");

			// Optional cross validation method - see https://mlnet-workshop.azurewebsites.net/posts/4.2-cross-val
			//var crossValidationResults = mlContext.Regression.CrossValidate(trainingData, trainingPipeline, numberOfFolds: 5);
			//var avgRSquared = crossValidationResults.Select(model => model.Metrics.RSquared).Average();
			//Console.WriteLine($"Cross Validated R-Squared: {avgRSquared}");

			// Save model
			Console.WriteLine("Saving model...");
			mlContext.Model.Save(model, trainingData.Schema, MODEL_SAVE_FILEPATH);
		}
    }
}
