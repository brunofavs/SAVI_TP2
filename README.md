<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]




<!-- PROJECT LOGO -->
<br />
<!-- 
  <a href="https://github.com/brunofavs/SAVI_TP1/graphs/">
    <img src="docs/LOGO.png" alt="Logo" width="550" height="350">
  </a> -->

<h3 align="center">Trabalho Prático 2</h3>

<h2><b> Repository Owner: Bruno Silva 98374
<br>Collaborators: Mário Vasconcelos 84081</b></h2>

  <p align="center">
    This repository was created for evaluation at Industrial Vision Advanced Systems "SAVI 23-24 Trabalho prático 2".
    <br />
    <!-- <a href="https://github.com/brunofavs/SAVI_TP1"><strong>Explore the Wiki »</strong></a> -->
    <br >
    <a href="https://github.com/brunofavs/SAVI_TP1/issues"> <u>Make Suggestion</u> </a>
  </p>
</div>
<br>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
     <li>
      <a href="#Objectives">Objectives</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#Setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
<br>



<!-- ABOUT THE PROJECT -->
## About The Project
<!-- <div align="center">
<img  src="docs/tracking.gif" alt="GIF animated" width="400">
</div>
<br> -->

This assignment was developed for Advanced Systems of Industrial Vision. It was a project to learn point cloud and image manipulation as well as classifier Neural Networks.

<div align="center">
<img  src="docs/guesses.png" alt="GUI" width="1000">
</div>
<br>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<div align="center">
<img  src="docs/scene.png" alt="Scene" width="1000">
</div>
<br>
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- Objectives -->
## Objectives

This project is capable of extracting objects from a Point Cloud of a scene, classify them and extract their features, such as height or color. The model was trained with the Washington Dataset, so it is better suited for classifying household objects.

<br>
<br>


<!-- GETTING STARTED -->
## Getting Started

Before anything, the user should set up a environment variable:

```
$SAVI_TP2  --> /path_to_project
```




## Setup
<h3><b>Libraries</b></h3>

To run the program, the following libraries should be installed:
```
opencv
torch
torchmetrics
vlc_python
gTTs
open3D
matplotlib
numpy
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

To run the project, execute the following script under `./src`

```
./main.py
```

Arguments when running `main.py`:
- -s -> Choose which scene to render;
- -pNN -> Enable Neural Network debugging mode.
- -us -> Use upscale for the NN inputs (takes a lot more time)

<!-- ### How it works -->







<!-- CONTACT -->
## Contact

Bruno Silva - bruno.favs@ua.pt

Mário Vasconcelos - mario.vasconcelos@ua.pt
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Professor Miguel Oliveira - mriem@ua.pt

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/RobutlerAlberto/RobutlerAlberto.svg?style=for-the-badge
[contributors-url]: https://github.com/brunofavs/SAVI_TP1/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/RobutlerAlberto/RobutlerAlberto.svg?style=for-the-badge
[forks-url]: https://github.com/RobutlerAlberto/RobutlerAlberto/network/members
[stars-shield]: https://img.shields.io/github/stars/RobutlerAlberto/RobutlerAlberto.svg?style=for-the-badge
[stars-url]: https://github.com/RobutlerAlberto/RobutlerAlberto/stargazers
[issues-shield]: https://img.shields.io/github/issues/RobutlerAlberto/RobutlerAlberto.svg?style=for-the-badge
[issues-url]: https://github.com/RobutlerAlberto/RobutlerAlberto/issues
[license-shield]: https://img.shields.io/github/license/RobutlerAlberto/RobutlerAlberto.svg?style=for-the-badge
[license-url]: https://github.com/RobutlerAlberto/RobutlerAlberto/blob/master/license.txt
[product-screenshot]: docs/logo.png
