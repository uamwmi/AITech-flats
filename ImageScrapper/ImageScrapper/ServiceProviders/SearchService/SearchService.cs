// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using ImageScrapper.ServiceProviders.WebsiteService;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace ImageScrapper.ServiceProviders.SearchService;

public class SearchService : ISearchService
{
    private readonly ILogger<SearchService> _log;
    private readonly IConfiguration _config;
    private readonly IWebsiteService _olxService;
    private readonly IWebsiteService _houzzService;

    public SearchService(ILogger<SearchService> log, IConfiguration config, IEnumerable<IWebsiteService> websiteServices)
    {
        _log = log;
        _config = config;
        websiteServices = websiteServices.ToList();
        _olxService = websiteServices.FirstOrDefault(x => x.GetType() == typeof(OlxWebsiteService)) ?? throw new MissingMemberException();
        _houzzService = websiteServices.FirstOrDefault(x => x.GetType() == typeof(HouzzWebsiteService)) ?? throw new MissingMemberException();
    }

    public async Task StartSearchService(string sourceWebsite = "")
    {
        _log.LogInformation("Searching started...");
        try
        {
            //await _olxService.StartScrapping();
            await _houzzService.StartScrapping();
        }
        catch (Exception e)
        {
            _log.LogError(e, "An error occurred while searching");
        }
        finally
        {
            _log.LogInformation("Searching completed");
        }
    }
}