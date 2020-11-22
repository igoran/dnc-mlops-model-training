using System;
using System.IO;
using System.Text;
using kernel;
using Microsoft.ML.Data;

namespace trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var buildConfig = BuildConfig.GetCurrent();

            PrintConfigValues(buildConfig);

            var dataFilePath = GetDataFilePath(buildConfig);

            Console.WriteLine(dataFilePath);

            var trainer = new TrainerForBinaryClassification(dataFilePath);

            var metrics = trainer.Evaluate();

            //  Ensure metrics are in a acceptable range
            if (metrics.F1Score < TrainerForBinaryClassification.Threshold)
            {
                throw new ApplicationException("F1Score is too low!");
            }

            var stats = GetModelStats(metrics);

            var modelOutputPath = buildConfig.GetModelFile().FullName;

            if (!trainer.SaveModel(modelOutputPath))
            {
                throw new ApplicationException($"ML Model cannot be saved to this position: {modelOutputPath}");
            }

            Console.WriteLine($"Model written to disk, location: {modelOutputPath}");

            var changeLogPath = buildConfig.GetChangeLog().FullName;

            File.WriteAllText(changeLogPath, stats);

            Console.WriteLine($"ChangeLog written to disk, location: {changeLogPath}");

        }

        private static string GetModelStats(BinaryClassificationMetrics metrics)
        {
            var sb = new StringBuilder();

            sb.AppendLine("Model quality metrics:");
            sb.AppendLine("--------------------------------");
            sb.AppendLine($"Accuracy: {metrics.Accuracy:P2}");
            sb.AppendLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            sb.AppendLine($"AUCPR: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            sb.AppendLine($"F1Score: {metrics.F1Score:P2}");
            sb.AppendLine($"Confusion Matrix:");
            sb.AppendLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            sb.AppendLine("--------------------------------");

            Console.WriteLine(sb);

            return sb.ToString();
        }

        private static string GetDataFilePath(BuildConfig buildConfig)
        {
            var dir = string.IsNullOrWhiteSpace(buildConfig.DataDirectory) ? Directory.GetCurrentDirectory() : buildConfig.DataDirectory;

            var path = Path.Combine(dir, "master_dataset.txt");

            if (!File.Exists(path)) Console.WriteLine($"File data not exists : {path}");

            return path;
        }

        private static void PrintConfigValues(BuildConfig buildConfig)
        {
            Console.WriteLine($"{nameof(buildConfig.Reason)} : {buildConfig.Reason}");

            Console.WriteLine($"{nameof(buildConfig.DataDirectory)} : {buildConfig.DataDirectory}");

            Console.WriteLine($"{nameof(buildConfig.OutputDirectory)} : {buildConfig.OutputDirectory}");
        }
    }

}
