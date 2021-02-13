using System;
using System.IO;
using kernel;
using Microsoft.ML;
using Shouldly;
using Xunit;
using Xunit.Abstractions;

namespace tester
{
    public class TrainerForBinaryClassificationFixture
    {
        private readonly ITestOutputHelper testOutputHelper;

        public TrainerForBinaryClassificationFixture(ITestOutputHelper testOutputHelper)
        {
            this.testOutputHelper = testOutputHelper;
            ModelFilePath = Path.Combine(Environment.CurrentDirectory, "Model.zip");
            DataFilePath = Path.Combine(Environment.CurrentDirectory, "training-data.tsv");
            Sut = new TrainerForBinaryClassification(DataFilePath);
        }

        public string ModelFilePath { get; set; }
        public string DataFilePath { get; set; }
        public TrainerForBinaryClassification Sut { get; set; }

        [Theory]
        [MemberData(nameof(FeedbackScenario.Inputs), MemberType = typeof(FeedbackScenario))]
        public void should_verify_prediction(string issue, bool expected)
        {
            // Act
            var sampleStatement = new SentimentIssue
            {
                SentimentText = issue
            };

            //  Arrange
            var result = Sut.Predict(sampleStatement);

            testOutputHelper.WriteLine(result.ToString());

            //  Assert
            result.Prediction.ShouldBe(expected);
        }

        [Theory]
        [MemberData(nameof(FeedbackScenario.Inputs), MemberType = typeof(FeedbackScenario))]
        public void should_verify_prediction_from_serialized_model(string issue, bool expected)
        {
            // Act
            var context = new MLContext();
            var sampleStatement = new SentimentIssue
            {
                SentimentText = issue
            };

            //  Arrange
            Sut.SaveModel(ModelFilePath);
            var transformer = context.Model.Load(ModelFilePath, out _);
            var predictionEngine = context.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(transformer);
            var result = predictionEngine.Predict(sampleStatement);

            testOutputHelper.WriteLine(result.ToString());

            //  Assert
            result.Prediction.ShouldBe(expected);
        }

        [Fact]
        public void should_print_metrics()
        {
            // Act
            var metrics = Sut.Evaluate();

            //  Arrange
            var text = Sut.ToTextStats(metrics);

            //  Assert
            text?.Length.ShouldBeGreaterThan(0);

            Console.WriteLine(Sut.ToTextStats(metrics));           
        }

        [Fact]
        public void should_metrics_be_in_range()
        {
            // Act

            //  Arrange
            var metrics = Sut.Evaluate();

            //  Assert
            metrics.Accuracy.ShouldBeInRange(TrainerForBinaryClassification.Threshold, 0.99f);
            metrics.AreaUnderRocCurve.ShouldBeInRange(TrainerForBinaryClassification.Threshold, 1);
            metrics.AreaUnderPrecisionRecallCurve.ShouldBeInRange(TrainerForBinaryClassification.Threshold, 1);
            metrics.F1Score.ShouldBeInRange(TrainerForBinaryClassification.Threshold, 1);
        }
    }
}
