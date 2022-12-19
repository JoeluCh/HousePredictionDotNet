using Microsoft.ML.Data;

namespace HousePrediction.Models
{
    public class HousePricePrediction
    {
        // Documentation: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-does-mldotnet-work
        // All algorithms also create new columns after they have performed a prediction.
        // The fixed names of these new columns depend on the type of machine learning algorithm.
        // For the regression task, one of the new columns is called Score.
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
