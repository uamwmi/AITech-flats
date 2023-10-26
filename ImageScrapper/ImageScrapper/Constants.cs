// SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
//
// SPDX-License-Identifier: MIT

using System.ComponentModel;
using System.Collections.Immutable;

namespace ImageScrapper;

public static class Constants
{
    public static class Olx
    {
        public const string Url =
            "https://www.olx.pl/d/nieruchomosci/mieszkania/sprzedaz/?search%5Bfilter_enum_market%5D%5B0%5D=secondary";
    }

    public static readonly ImmutableSortedDictionary<string, string> HouzzUrls = new Dictionary<string, string>()
    {
        {"Industrial", "https://www.houzz.com/photos/industrial-living-room-ideas-phbr1-bp~t_718~s_2113"},
        {"Rustic", "https://www.houzz.com/photos/rustic-living-room-ideas-and-designs-phbr1-bp~t_718~s_2111"},
        {"Modern", "https://www.houzz.com/photos/modern-living-room-ideas-phbr1-bp~t_718~s_2105"},
        {"Scandinavian", "https://www.houzz.com/photos/scandinavian-living-room-ideas-phbr1-bp~t_718~s_22848"},
        {"Classic", "https://www.houzz.com/photos/traditional-living-room-ideas-phbr1-bp~t_718~s_2107"},
        {"ArtDeco", "https://www.houzz.com/photos/query/art-deco-living-room/nqrwns"},
        {"Vintage", "https://www.houzz.com/photos/query/vintage-living-room/nqrwns"},
        {"Glamour", "https://www.houzz.com/photos/query/glamour-living-room/nqrwns"},
        {"Minimalistic", "https://www.houzz.com/photos/query/minimalist-living-room/nqrwns"},
    }.ToImmutableSortedDictionary();

    public static class RestSharp
    {
        public const string Cookies =
            "laquesis=buy-2895@b#decision-196@a#er-1724@b#er-1892@a#f8nrp-1268@b#jobs-3717@c#jobs-3834@a#jobs-4023@a#jobs-4078@b#jobs-4134@a#jobs-4425@c#nhub-27@b#oesx-2285@b#olxeu-40144@b; laquesisff=a2b-000#aut-388#aut-716#buy-2279#buy-2489#buy-2811#dat-2874#decision-256#do-2963#euonb-114#euonb-48#grw-124#kuna-307#kuna-314#kuna-554#kuna-603#mart-555#mou-1052#oesx-1437#oesx-1643#oesx-645#oesx-867#olxeu-0000#olxeu-29763#psm-308#psm-402#psm-457#psm-574#sd-570#srt-1289#srt-1346#srt-1434#srt-1593#srt-1758#srt-474#srt-475#srt-683#srt-899; lqstatus=1669247565";
    }
}