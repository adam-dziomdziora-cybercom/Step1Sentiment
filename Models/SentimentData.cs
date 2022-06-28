using Microsoft.ML.Data;

namespace Step1Sentiment.Models {
    public class SentimentData {
        [LoadColumn (0)]
        public string SentimentText=string.Empty;

        [LoadColumn (1), ColumnName ("Label")]
        public bool Sentiment;
    }
}