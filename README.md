<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

<div align="center"><a title="" href="https://github.com/zjykzj/anchor-boxes"><img align="center" src="./imgs/anchor-boxes.png" alt=""></a></div>

<p align="center">
  «anchor-boxes» generated the anchor-boxes required for training YOLO networks
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

* For Darknet implemetation, see [darknet](darknet)
  ![](./assets/canva_darknet.jpg)

* For custom YOLOv2 anchor-boxes implementation, see [v2](v2)
  ![](./assets/canva_v2.jpg)

* For custom YOLOv3 anchor-boxes implementation, see [v3](v3)
  ![](./assets/canva_v3.jpg)

* ***Blank boxes represent anchors based on VOC***
* ***Grayscale boxes represent anchors based on COCO***

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

This repository implements different ways of generating code for anchor-boxes, including YOLOv2/YOLOv3 anchor-box generation, using PASCAL VOC and COCO datasets

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/tree/master)
* [lars76/kmeans-anchor-boxes](https://github.com/lars76/kmeans-anchor-boxes)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/anchor-boxes/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj