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

            var trainer = new TrainerForBinaryClassification(dataFilePath);

            var metrics = trainer.Evaluate();

            //  Ensure metrics are in a acceptable range
            if (metrics.F1Score < TrainerForBinaryClassification.Threshold)
            {
                throw new ApplicationException("F1Score is too low!");
            }

            var modelOutputPath = buildConfig.GetModelFile().FullName;

            if (!trainer.SaveModel(modelOutputPath))
            {
                throw new ApplicationException($"ML Model cannot be saved to this position: {modelOutputPath}");
            }

            var printableMetrics = trainer.ToTextStats(metrics);

            Console.WriteLine(printableMetrics);

            Console.WriteLine($"Model written to disk, location: {modelOutputPath}");

            var changeLogPath = buildConfig.GetReleaseInfoMarkdown().FullName;

            File.WriteAllText(changeLogPath, trainer.ToMarkDownStats(metrics));

            Console.WriteLine($"Markdown changeLog written to disk, location: {changeLogPath}");

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
