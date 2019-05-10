using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Step1Sentiment.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace Step1Sentiment {
    class Program {

        static readonly string[] _dataNames = { "yelp_labelled.txt", "imdb_labelled.txt", "amazon_labelled.txt" };
        static readonly string _dataPath = Path.Combine (Environment.CurrentDirectory, "Data", _dataNames[0]);
        static readonly string _modelPath = Path.Combine (Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main (string[] args) {
            MLContext mlContext = new MLContext ();
            TrainTestData splitDataView = LoadData (mlContext);
            ITransformer model = BuildAndTrainModel (mlContext, splitDataView.TrainSet);
            Evaluate (mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem (mlContext, model);
            UseModelWithBatchItems(mlContext,model);
        }

        public static TrainTestData LoadData (MLContext mlContext) {
            // Load from text file
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData> (_dataPath, hasHeader : false);

            // Split for test and train sets
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit (dataView, testFraction : 0.2);

            // DEBUG the splitted data
            var debugTrainSet = splitDataView.TrainSet.Preview (maxRows: 1000);
            var debugTestSet = splitDataView.TestSet.Preview (maxRows: 1000);

            var testSetCount = debugTestSet.RowView.Count ();
            var trainSetCount = debugTrainSet.RowView.Count ();

            Console.WriteLine ($"Training set: {trainSetCount}, test set: {testSetCount}");
            //

            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel (MLContext mlContext, IDataView splitTrainSet) {
            var estimator = mlContext.Transforms.Text
                .FeaturizeText (
                    outputColumnName: "Features",
                    inputColumnName : nameof (SentimentData.SentimentText))
                .Append (mlContext.BinaryClassification.Trainers
                    .SdcaLogisticRegression (
                        labelColumnName: "Label",
                        featureColumnName: "Features")
                );
            Console.WriteLine ("=============== Create and Train the Model ===============");

            Stopwatch stopWatch = new Stopwatch ();
            stopWatch.Start ();
            var model = estimator.Fit (splitTrainSet);
            stopWatch.Stop ();
            Console.WriteLine ($"=============== End of training, taken {stopWatch.ElapsedMilliseconds}ms ===============");
            return model;
        }

        public static void Evaluate (MLContext mlContext, ITransformer model, IDataView splitTestSet) {
            Console.WriteLine ("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform (splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate (predictions, "Label");
            Console.WriteLine ();
            Console.WriteLine ("Model quality metrics evaluation");
            Console.WriteLine ("--------------------------------");
            Console.WriteLine ($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine ($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine ($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine ("=============== End of model evaluation ===============");
        }

        private static void UseModelWithSingleItem (MLContext mlContext, ITransformer model) {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction =
                mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction> (model);

            SentimentData sampleStatement = new SentimentData {
                SentimentText = "this is outstading training with an amazing trainer"
            };

            var resultprediction = predictionFunction.Predict (sampleStatement);
            Console.WriteLine ();
            Console.WriteLine ("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine ();
            Console.WriteLine ($"Sentiment: {resultprediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");

            Console.WriteLine ("=============== End of Predictions ===============");
            Console.WriteLine ();
        }

        public static void UseModelWithBatchItems (MLContext mlContext, ITransformer model) {
            SentimentData[] sentiments = new [] {
                new SentimentData {
                SentimentText = "This was a horrible meal"
                },
                new SentimentData {
                SentimentText = "I love this spaghetti."
                }
            };
            IDataView batchComments = mlContext.Data.LoadFromEnumerable (sentiments);

            IDataView predictions = model.Transform (batchComments);
            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data
                .CreateEnumerable<SentimentPrediction> (predictions, reuseRowObject : false);

            Console.WriteLine ();

            Console.WriteLine ("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults) {
                Console.WriteLine ($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine ("=============== End of predictions ===============");
        }
    }
}