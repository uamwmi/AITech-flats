// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

﻿using ImageScrapper.ServiceProviders.SearchService;
using ImageScrapper.ServiceProviders.WebsiteService;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using RestSharp;
using Serilog;

namespace ImageScrapper;

public class Program
{
    public static async Task Main(string[] args)
    {
        var builder = new ConfigurationBuilder();
        BuildConfig(builder);
        ConfigureLogger(builder);
        var host = CreateHost();

        var searchService = ActivatorUtilities.CreateInstance<SearchService>(host.Services);
        await searchService.StartSearchService();
    }

    public static void ConfigureLogger(ConfigurationBuilder builder)
    {
        var time = DateTime.Now.ToString("vyyyy-MM-dd__HH-mm");
        var config = builder.Build();
        Log.Logger = new LoggerConfiguration()
            .ReadFrom.Configuration(config)
            .Enrich.FromLogContext()
            .WriteTo.Console()
            .WriteTo.File(config["Path:LogPath"] + $"log-{time}.txt")
            .CreateLogger();
    }

    public static IHost CreateHost()
    {
        return Host.CreateDefaultBuilder()
            .ConfigureServices((_, services) =>
            {
                services.AddTransient<IWebsiteService, OlxWebsiteService>();
                services.AddTransient<IWebsiteService, HouzzWebsiteService>();
                services.AddTransient<ISearchService, SearchService>();
                services.AddSingleton(_ => new RestClient());
                services.AddHttpClient();
                // services.AddHttpClient<IWebsiteService, OlxWebsiteService>(client =>
                // {
                //     client.DefaultRequestHeaders.Add("User-Agent", "PostmanRuntime/7.29.2");
                //     client.DefaultRequestHeaders.Add("Accep-Encoding", "gzip, deflate, br");
                //     client.DefaultRequestHeaders.Add("Accept", "*/*");
                //     client.DefaultRequestHeaders.Add("Connection", "keep-alive");
                // });
            })
            .UseSerilog()
            .Build();
    }


    public static void BuildConfig(IConfigurationBuilder builder)
    {
        builder.SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .AddEnvironmentVariables();
    }
}