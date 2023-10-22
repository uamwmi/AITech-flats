// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

namespace ImageScrapper.ServiceProviders.SearchService;

public interface ISearchService
{
    /// <summary>
    /// Starts Search Service for given source website.
    /// </summary>
    /// <param name="sourceWebsite">Optional param for sourceService.</param>
    Task StartSearchService(string sourceWebsite = "");

}