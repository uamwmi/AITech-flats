// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using Newtonsoft.Json;

namespace ImageScrapper.Models.Olx;

    public class OlxWebsiteOffer
{
    [JsonProperty("id")]
    public int id { get; set; }

    [JsonProperty("title")]
    public string title { get; set; }

    [JsonProperty("description")]
    public string description { get; set; }

    [JsonProperty("url")]
    public string url { get; set; }

    [JsonProperty("createdTime")]
    public DateTime createdTime { get; set; }

    [JsonProperty("lastRefreshTime")]
    public DateTime lastRefreshTime { get; set; }

    [JsonProperty("pushupTime")]
    public DateTime pushupTime { get; set; }

    [JsonProperty("isActive")]
    public bool isActive { get; set; }

    [JsonProperty("status")]
    public string status { get; set; }

    [JsonProperty("params")]
    public List<Param> @params { get; set; }

    [JsonProperty("itemCondition")]
    public string itemCondition { get; set; }

    [JsonProperty("price")]
    public Price price { get; set; }

    [JsonProperty("photos")]
    public List<string> photos { get; set; }

    [JsonProperty("photosSet")]
    public List<string> photosSet { get; set; }

    [JsonProperty("location")]
    public Location location { get; set; }

    [JsonProperty("urlPath")]
    public string urlPath { get; set; }
}

    public class Location
    {
        [JsonProperty("cityName")]
        public string cityName { get; set; }

        [JsonProperty("cityId")]
        public int cityId { get; set; }

        [JsonProperty("cityNormalizedName")]
        public string cityNormalizedName { get; set; }

        [JsonProperty("regionName")]
        public string regionName { get; set; }

        [JsonProperty("regionId")]
        public int regionId { get; set; }

        [JsonProperty("regionNormalizedName")]
        public string regionNormalizedName { get; set; }

        [JsonProperty("districtName")]
        public object districtName { get; set; }

        [JsonProperty("districtId")]
        public int districtId { get; set; }

        [JsonProperty("pathName")]
        public string pathName { get; set; }
    }

    public class Param
    {
        [JsonProperty("key")]
        public string key { get; set; }

        [JsonProperty("name")]
        public string name { get; set; }

        [JsonProperty("type")]
        public string type { get; set; }

        [JsonProperty("value")]
        public string value { get; set; }

        [JsonProperty("normalizedValue")]
        public string normalizedValue { get; set; }
    }

    public class Price
    {
        [JsonProperty("budget")]
        public bool budget { get; set; }

        [JsonProperty("free")]
        public bool free { get; set; }

        [JsonProperty("exchange")]
        public bool exchange { get; set; }

        [JsonProperty("displayValue")]
        public string displayValue { get; set; }

        [JsonProperty("regularPrice")]
        public RegularPrice regularPrice { get; set; }
    }

    public class PriceFormatConfig
    {
        [JsonProperty("decimalSeparator")]
        public string decimalSeparator { get; set; }

        [JsonProperty("thousandsSeparator")]
        public string thousandsSeparator { get; set; }
    }

    public class RegularPrice
    {
        [JsonProperty("value")]
        public int value { get; set; }

        [JsonProperty("currencyCode")]
        public string currencyCode { get; set; }

        [JsonProperty("currencySymbol")]
        public string currencySymbol { get; set; }

        [JsonProperty("negotiable")]
        public bool negotiable { get; set; }

        [JsonProperty("priceFormatConfig")]
        public PriceFormatConfig priceFormatConfig { get; set; }
    }