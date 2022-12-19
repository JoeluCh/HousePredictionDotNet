using HousePrediction.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

MLContext mlContext = new();

// 1. Import or create training data
IDataView trainingData = mlContext.Data.LoadFromEnumerable(new HouseData[]
{
    new HouseData() { Size = 1.1F, Price = 1.2F },
    new HouseData() { Size = 1.9F, Price = 2.3F },
    new HouseData() { Size = 2.8F, Price = 3.0F },
    new HouseData() { Size = 3.4F, Price = 3.7F },
});

// 2. Specify data preparation 
EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> pipeline = mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: new[] { "Size" }).Append(estimator: mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

// 3. Train model
TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> model = pipeline.Fit(trainingData);

// 4. Save the model for future use
mlContext.Model.Save(model, trainingData.Schema, @"D:\ML\HousePrediction\Trained\trainedModel.zip");



