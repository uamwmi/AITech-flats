// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

namespace ResellScrapperV3.Models;

public interface IWebsiteConfig
{
    string Category { get; set; }
    string Gender { get; set; }
    string Brand { get; set; }
}