using Microsoft.ML.Data;

namespace Step1Sentiment.Models {
    public class SentimentPrediction : SentimentData {
        [ColumnName ("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}