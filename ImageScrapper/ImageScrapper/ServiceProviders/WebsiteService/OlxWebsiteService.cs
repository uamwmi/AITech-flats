// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using System.Net;
using System.Reflection;
using System.Text.RegularExpressions;
using HtmlAgilityPack;
using ImageScrapper.Models.Olx;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using RestSharp;

// using ImageScrapper.Exceptions;
// using ImageScrapper.Mappings;
// using ImageScrapper.Models.Allegro;

namespace ImageScrapper.ServiceProviders.WebsiteService;

public class OlxWebsiteService : IWebsiteService
{
    private readonly ILogger<OlxWebsiteService> _log;
    private readonly IConfiguration _config;
    private readonly RestClient _httpRestClient;

    public OlxWebsiteService(ILogger<OlxWebsiteService> log, IConfiguration config, RestClient httpRestClient)
    {
        _httpRestClient = httpRestClient;
        _log = log;
        _config = config.GetSection("Olx");
    }


    public async Task StartScrapping()
    {
        var startPage = int.Parse(_config["StartPageNumber"]!);

        for (var i = startPage; i < 50; i++)
        {
            _log.LogInformation("Starting scrapping page {@Page}...", i);

            try
            {
                var url = BuildUrl(i);
                var productList = await GetProductList(url);
                var olxWebsiteProductList =
                    JsonConvert.DeserializeObject<List<OlxWebsiteOffer>>(JsonConvert.SerializeObject(productList));

                await DownloadImages(olxWebsiteProductList);
            }
            catch(Exception ex)
            {
                _log.LogError(exception: ex, "Error while scrapping page {@Page}...", i);
            }
            finally
            {
                _log.LogInformation("Ended scrapping page {@Page}...", i);
            }
        }
    }

    public async Task<IList<object>> GetProductList(string websiteUrl)
    {
        var urlResponse = await GetAsync(websiteUrl);

        var content = urlResponse.Content;
        var doc = new HtmlDocument();
        doc.LoadHtml(content);

        return ExtractProductList(doc);
    }


    // Private
    private string BuildUrl(int page)
    {
        return $"{Constants.Olx.Url}&page={page}";
    }

    private List<object> ExtractProductList(HtmlDocument doc)
    {
        var scriptNode = doc.DocumentNode.SelectNodes("//script[@id='olx-init-config']");
        var split = scriptNode.First().InnerHtml.Split("window.__PRERENDERED_STATE__= \"")[1].Split("\";")[0];
        var jsonString = Regex.Unescape(split);
        var jsonObject = JsonConvert.DeserializeObject<dynamic>(jsonString);
        var offers = jsonObject!.listing.listing.ads;

        return offers.ToObject<List<object>>();
    }

    private async Task<RestResponse> GetAsync(string url)
    {
        var request = new RestRequest(url);
        request.AddHeader("Accept", "application/json");
        request.AddHeader("Cookie", Constants.RestSharp.Cookies);
        var response = await _httpRestClient.ExecuteAsync(request);

        if (!response.IsSuccessful)
        {
            throw new HttpRequestException($"Error while getting url {url}");
        }

        return response;
    }

    private async Task DownloadImages(List<OlxWebsiteOffer> websiteProductList)
    {
        var imagesTuple = websiteProductList
            .Select(i => new Tuple<string, string, List<string>>(i.id.ToString(), i.location.regionNormalizedName, i.photos))
            .ToList();
        imagesTuple.Sort();

        _log.LogInformation("OLX Website Offers info: {@OlxWebsiteOffers}", websiteProductList.ToDictionary(o => o.id.ToString(), o => o.url));

        foreach (var image in imagesTuple)
        {
            var imageUrls = image.Item3;
            var i = 0;

            using WebClient client = new();
            foreach (var imageUrl in imageUrls)
            {
                i++;
                var imageParentDirectory = PrepareDirectory(image);
                var imageName = $"{image.Item1}-{i}.jpg";
                var imageDirectory = Path.Combine(imageParentDirectory, imageName);

                //client.DownloadFileAsync(new Uri(imageUrl), imageDirectory);
                if (File.Exists(imageDirectory))
                {
                    _log.LogInformation("Skipping image {@ImageDirectory} because it already exists", imageDirectory);
                    continue;
                }

                await client.DownloadFileTaskAsync(new Uri(imageUrl), imageDirectory);
                _log.LogInformation("Downloading image to {@ImageDirectory}",imageDirectory);
            }

            _log.LogInformation("Downloaded: {TasksCount} images", i);
        }
    }

    private string PrepareDirectory(Tuple<string, string, List<string>> imageTuple)
    {
        var projectPath = _config["Path:ProjectPath"] ?? throw new KeyNotFoundException("ProjectPath not found");
        var imageDirectory = Path.Combine(projectPath, "Images", imageTuple.Item2);

        if (!Directory.Exists(imageDirectory))
            Directory.CreateDirectory(imageDirectory);

        return imageDirectory;
    }
}