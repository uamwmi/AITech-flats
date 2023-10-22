// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using System.Globalization;
using System.Net;
using System.Text.RegularExpressions;
using HtmlAgilityPack;
using ImageScrapper.Models.Olx;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using RestSharp;

namespace ImageScrapper.ServiceProviders.WebsiteService;

public class HouzzWebsiteService : IWebsiteService
{
    private readonly ILogger<HouzzWebsiteService> _log;
    private readonly IConfiguration _config;
    private readonly RestClient _httpRestClient;
    private readonly HttpClient _httpClient;

    public HouzzWebsiteService(ILogger<HouzzWebsiteService> log, IConfiguration config, RestClient httpRestClient, HttpClient httpClient)
    {
        _httpRestClient = httpRestClient;
        _httpClient = httpClient;
        _log = log;
        _config = config;
    }


    public async Task StartScrapping()
    {
        var startPage = int.Parse(_config["Houzz:StartPageNumber"]!);
        var endPage = int.Parse(_config["Houzz:EndPageNumber"]!);

        foreach (var style in Constants.HouzzUrls)
        {
            _log.LogInformation("Starting scrapping style: {@Style}...", style.Key);

            for (var i = startPage; i <= endPage; i++)
            {
                try
                {
                    var url = BuildUrl(i, style.Value);
                    _log.LogInformation("Starting scrapping page: {@Page}...", url);
                    var imageUrlList = await GetProductList(url);
                    await DownloadImages(imageUrlList, style.Key);
                }
                catch (Exception e)
                {
                    _log.LogError(e, "Error while scrapping style: {@Style} at page: {@Page}", style.Key, i);
                }
                finally
                {
                    _log.LogInformation("Finished scrapping style: {@Style} at page: {@Page}", style.Key, i);
                }
            }
        }
    }

    public async Task<IList<object>> GetProductList(string websiteUrl)
    {
        var urlResponse = await GetAsync(websiteUrl);

        //var content = urlResponse.Content;
        var content = await urlResponse.Content.ReadAsStringAsync();
        var doc = new HtmlDocument();
        doc.LoadHtml(content);

        var imageList = ExtractImageList(doc);
        _log.LogInformation("Found {@Count} images on page {@Page}", imageList.Count, websiteUrl);

        return new List<object>(imageList);
    }


    // Private

    private List<string> ExtractImageList(HtmlDocument doc)
    {
        var pattern = @"https.*.jpg";
        try
        {
            return doc.DocumentNode
                .SelectNodes("//img[@class='hz-photo-card__img']")
                .Select(n =>
                    Regex.Match(n.GetAttributes().First(a => a.Name == "srcset").Value.Split(',').ElementAt(1), pattern)
                        .Value
                )
                .ToList();
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Error while extracting image list");
            throw;
        }
    }

    private async Task DownloadImages(IEnumerable<object> imageList, string style)
    {
        var imageListString = imageList.Cast<string>().ToList();
        var i = 0;
        var pattern = @"~?(.{16}_.*-w720-h720)";

        using WebClient client = new();
        foreach (var image in imageListString)
        {
            var imgName = image.Split('/').Last();
            //imgName = $"{Regex.Match(imgName, @"~?(.{16})_").Groups[1].Value}.jpg";
            imgName = $"{Regex.Match(imgName, pattern).Groups[1].Value}.jpg";
            var imageParentDirectory = PrepareDirectory(style);
            var imageDirectory = Path.Combine(imageParentDirectory, imgName);

            if (File.Exists(imageDirectory))
            {
                _log.LogInformation("Skipping image {@ImageDirectory} because it already exists", imageDirectory);
                continue;
            }

            await client.DownloadFileTaskAsync(new Uri(image), imageDirectory);
            i++;
            _log.LogInformation("Downloading image to {@ImageDirectory}", imageDirectory);
        }

        _log.LogInformation("Downloaded: {TasksCount} images", i);
    }

    private async Task<HttpResponseMessage> GetAsync(string url)
    {
        //var request = new RestRequest(url);
        //request.AddHeader("Accept", "application/json");
        //var response = await _httpRestClient.ExecuteAsync(request);

        var response = await _httpClient.GetAsync(url);

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException($"Error while getting url {url}");
        }

        return response;
    }

    private string PrepareDirectory(string style)
    {
        var projectPath = _config["Path:ProjectPath"] ?? throw new KeyNotFoundException("ProjectPath not found");
        var styleDirectory = Path.Combine(projectPath, "Images", "Houzz", style);

        if (!Directory.Exists(styleDirectory))
            Directory.CreateDirectory(styleDirectory);

        return styleDirectory;
    }

    private string BuildUrl(int page, string url)
    {
        return url.Contains("query")
            ? $"{url}/p/{page * 20}"
            : $"{url}?pg={page}";
    }
}