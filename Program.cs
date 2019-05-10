using System;
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

    }
}