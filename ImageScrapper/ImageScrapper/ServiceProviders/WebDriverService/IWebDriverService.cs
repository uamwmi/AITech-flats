// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using HtmlAgilityPack;

namespace ImageScrapper.ServiceProviders.WebDriverService;

public interface IWebDriverService
{
    HtmlDocument GetPage(string pageUrl);
}