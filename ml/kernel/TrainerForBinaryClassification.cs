using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace kernel
{
    public class TrainerForBinaryClassification
    {
        private readonly MLContext context;
        private readonly IDataView dataView;
        public const float Threshold = 0.70f;

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
            .Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression());
            //.Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

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

        public string ToTextStats(BinaryClassificationMetrics metrics)
        {
            var sb = new StringBuilder();

            sb.AppendLine("Model quality metrics:");
            sb.AppendLine("--------------------------------");
            sb.AppendLine("Learners: LbfgsLogisticRegression");
            sb.AppendLine("--------------------------------");
            sb.AppendLine($"Accuracy: {metrics.Accuracy:P2}");
            sb.AppendLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            sb.AppendLine($"AUCPR: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            sb.AppendLine($"F1Score: {metrics.F1Score:P2}");
            sb.AppendLine($"Confusion Matrix:");
            sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            sb.AppendLine("--------------------------------");

            return sb.ToString();
        }

        public string ToMarkDownStats(BinaryClassificationMetrics metrics)
        {
            var sb = new StringBuilder();

            sb.AppendLine("# Model Quality Metrics:");
            sb.AppendLine("##Learner: LbfgsLogisticRegression");
            sb.AppendLine("| Parameter | Value |");
            sb.AppendLine("| :---      |    :----: |");
            sb.AppendLine($"| Accuracy | **{metrics.Accuracy:P2}**|");
            sb.AppendLine($"| AUC | **{metrics.AreaUnderRocCurve:P2}**|");
            sb.AppendLine($"| AUCPR | **{metrics.AreaUnderPrecisionRecallCurve:P2}**|");
            sb.AppendLine($"| F1Score | **{metrics.F1Score:P2}**|");


            sb.AppendLine($"---");

            sb.AppendLine($"## Confusion Matrix:");

            sb.AppendLine($"```");
            sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            sb.AppendLine($"```");

            return sb.ToString();
        }
    }
}
