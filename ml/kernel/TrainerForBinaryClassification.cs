using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace kernel
{
    public class TrainerForBinaryClassification
    {
        private readonly MLContext context;
        private readonly IDataView dataView;
        public const float Threshold = 0.75f;

        public TrainerForBinaryClassification(string datasetPath)
        {
            if (string.IsNullOrWhiteSpace(datasetPath))
                throw new ArgumentException("Value cannot be null or whitespace.", nameof(datasetPath));

            context = new MLContext();

            // Step 1. Read in the input data for model training
            dataView = context.Data.LoadFromTextFile<SentimentIssue>(datasetPath);

            Init();
        }

        private void Init()
        {
            var trainTestSplit = context.Data.TrainTestSplit(dataView, testFraction: 0.2);
            DataForTraining = trainTestSplit.TrainSet; // 80% for training purpose
            DataForTesting = trainTestSplit.TestSet;   // 20% for testing purpose
        }

        private Lazy<ITransformer> BuildAndTrainModel => new Lazy<ITransformer>(() =>
        {
            var dataPipeline = context.Transforms.Text.FeaturizeText("Features",nameof(SentimentIssue.SentimentText));

            // Step 3. Build your estimator
            var trainingPipeline = dataPipeline
            //.Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression())
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            // Step 4. Train your Model
            return trainingPipeline.Fit(DataForTraining);
        });

        public BinaryClassificationMetrics Evaluate(IDataView samples = default)
        {
            //https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/metrics

            var predictions = BuildAndTrainModel.Value.Transform(samples ?? DataForTesting);

            return context.BinaryClassification.Evaluate(predictions);
        }

        public bool SaveModel(string filePath)
        {
            context.Model.Save(BuildAndTrainModel.Value, dataView.Schema, filePath);

            return File.Exists(filePath);
        }

        public SentimentPrediction Predict(SentimentIssue issue)
        {
            var predictionEngine = context.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(BuildAndTrainModel.Value);

            return predictionEngine.Predict(issue);
        }

        public IDataView DataForTraining { get; private set; }

        public IDataView DataForTesting { get; private set; }
    }
}
