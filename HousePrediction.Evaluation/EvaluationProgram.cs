using HousePrediction.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

MLContext mlContext = new();
ITransformer trainedModel = mlContext.Model.Load(@"D:\ML\HousePrediction\Trained\trainedModel.zip", out _);

#region Model Evaluation
IDataView evHouseDataView = mlContext.Data.LoadFromEnumerable(new HouseData[]
{
    new HouseData() { Size = 1.1F, Price = 0.98F }, // 1,100 ft^2 ; 0.98
    new HouseData() { Size = 1.9F, Price = 2.1F },
    new HouseData() { Size = 2.8F, Price = 2.9F },
    new HouseData() { Size = 3.4F, Price = 3.6F },
});

IDataView evPriceDataView = trainedModel.Transform(input: evHouseDataView);
RegressionMetrics metrics = mlContext.Regression.Evaluate(evPriceDataView, labelColumnName: "Price");

// these values will determine the performance of the model
Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");
#endregion

#region Prediction
// define input and output models
PredictionEngine<HouseData, HousePricePrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(trainedModel);

// make predictions, use the model for what's been created
while (true)
{
    Console.WriteLine();
    Console.Write("Value to predict in squared feet / 1000: ");
    string reading = Console.ReadLine();

    if (float.TryParse(reading, out float newSizeValue) is false)
    {
        Console.WriteLine($"Input '{reading}' is not a valid float number.");
        continue;
    }

    HousePricePrediction predictedModel = predictionEngine.Predict(new HouseData() { Size = newSizeValue });
    Console.WriteLine($"Predicted price for size: {newSizeValue * 1000} sq ft = {predictedModel.Price * 100:C}k");
}
#endregion