<!--
SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>

SPDX-License-Identifier: MIT
-->

# Image scrapper console application

This is a simple console application that scrapes images from web services.

## Sites supported

* [x] [Olx](https://www.olx.pl/)
* [x] [Houzz](https://www.houzz.com/)

## Image details

* Houzz images are size in (900,900)

## Instruction

* Clone the repository
* Open the solution in Visual Studio/Rider
* Update the `appsettings.json` file with the desired configuration
* Run the application: ImageScrapper
* .NET 6.0 is required

## Images structure

```text
project
│   README.md
└───Images
    └───Olx
    │   └───wielkopolskie
    │   │      1.jpg
    │   │      2.jpg
    │   │      ...
    │   └───zachodnipomorskie
    │          1.jpg
    │          2.jpg
    │          ...          
    └───Houzz
        └───scandinavian 
        │      1.jpg
        │      2.jpg
        │      ...
        └───modern
               1.jpg
               2.jpg
               ... 
```
