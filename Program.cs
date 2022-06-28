// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Step1Sentiment.Models;
using System.Diagnostics;
using static Microsoft.ML.DataOperationsCatalog;

Console.WriteLine("Hello, World!");

var _dataNames = new[] { "yelp_labelled.txt", "imdb_labelled.txt", "amazon_labelled.txt" };
var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", _dataNames[0]);


var mlContext = new MLContext();
TrainTestData splitDataView = LoadData(mlContext, dataPath);
ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
Evaluate(mlContext, model, splitDataView.TestSet);
UseModelWithSingleItem(mlContext, model);
UseModelWithBatchItems(mlContext, model);


static TrainTestData LoadData(MLContext mlContext, string dataPath)
{
    // Load from text file
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: false);

    // Split for test and train sets
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

    // DEBUG the splitted data
    var debugTrainSet = splitDataView.TrainSet.Preview(maxRows: 1000);
    var debugTestSet = splitDataView.TestSet.Preview(maxRows: 1000);

    var testSetCount = debugTestSet.RowView.Length;
    var trainSetCount = debugTrainSet.RowView.Length;

    Console.WriteLine($"Training set: {trainSetCount}, test set: {testSetCount}");
    //

    return splitDataView;
}

static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text
        .FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features")
        );
    Console.WriteLine("=============== Create and Train the Model ===============");

    var stopWatch = new Stopwatch();
    stopWatch.Start();
    var model = estimator.Fit(splitTrainSet);
    stopWatch.Stop();
    Console.WriteLine($"=============== End of training, taken {stopWatch.ElapsedMilliseconds}ms ===============");
    return model;
}

static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction =
        mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    var sampleStatement = new SentimentData
    {
        SentimentText = "this is outstading training with an amazing trainer"
    };

    var resultprediction = predictionFunction.Predict(sampleStatement);
    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultprediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultprediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    var sentiments = new[] {
                new SentimentData {
                SentimentText = "This was a horrible meal"
                },
                new SentimentData {
                SentimentText = "I love this spaghetti."
                }
            };
    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    IDataView predictions = model.Transform(batchComments);
    // Use model to predict whether comment data is Positive (1) or Negative (0).
    IEnumerable<SentimentPrediction> predictedResults = mlContext.Data
        .CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

    }
    Console.WriteLine("=============== End of predictions ===============");
}