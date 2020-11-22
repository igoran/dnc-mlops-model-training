﻿using Microsoft.ML.Data;

namespace kernel
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }
}
