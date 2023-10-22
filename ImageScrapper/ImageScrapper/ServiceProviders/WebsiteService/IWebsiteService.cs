// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

namespace ImageScrapper.ServiceProviders.WebsiteService;

public interface IWebsiteService
{
    Task<IList<object>> GetProductList(string websiteUrl);
    Task StartScrapping();
}