using System;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace trainer
{
    public class BuildConfig
    {
        public string Reason { get; set; }
        public string OutputDirectory { get; set; }
        public string DataDirectory { get; set; }

        public FileInfo GetModelFile()
        {
            var dir = string.IsNullOrWhiteSpace(DataDirectory) ? Environment.CurrentDirectory : OutputDirectory;

            return new FileInfo(Path.Combine(dir, "model.zip"));
        }

        public FileInfo GetChangeLog()
        {
            var dir = string.IsNullOrWhiteSpace(DataDirectory) ? Environment.CurrentDirectory : OutputDirectory;

            return new FileInfo(Path.Combine(dir, "CHANGELOG.txt"));
        }

        public FileInfo GetReleaseInfoMarkdown()
        {
            var dir = string.IsNullOrWhiteSpace(DataDirectory) ? Environment.CurrentDirectory : OutputDirectory;

            return new FileInfo(Path.Combine(dir, "ReleaseInfo.md"));
        }

        private static readonly Lazy<IServiceProvider> _serviceProvider = new Lazy<IServiceProvider>(() =>
        {
            var config = new ConfigurationBuilder().AddEnvironmentVariables().Build();
            var services = new ServiceCollection();
            var buildConfig = new BuildConfig();
            config.Bind("Build", buildConfig);
            services.AddSingleton(buildConfig);
            services.AddSingleton<IConfiguration>(config);
            return services.BuildServiceProvider();
        });

        public static BuildConfig GetCurrent() => _serviceProvider.Value.GetRequiredService<BuildConfig>();
    }
}
