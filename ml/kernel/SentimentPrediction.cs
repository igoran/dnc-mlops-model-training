using Microsoft.ML.Data;

namespace kernel
{
    public class SentimentPrediction : SentimentIssue
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

        public override string ToString()
        {
            return $"\"{SentimentText}\" is '{(Prediction ? "Positive":"Negative")}' Score: {Score} Probability: {Probability}";
        }
    }
}
