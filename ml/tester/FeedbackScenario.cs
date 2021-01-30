using System.Collections.Generic;

namespace tester
{
    public class FeedbackScenario
    {
        public static IEnumerable<object[]> Inputs =>
            new List<object[]>
            {
                new object[] {"I loving DevOps Heroes 2019!", true},
                new object[] {"I highly recommend this session", true},
                new object[]{"Not enjoyed, I don't recommend this speech",false},
                new object[] {"This session and that speakers are fantastic", true},
            };
    }
}
