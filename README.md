# Periphoscape

## About this repository

This repository provides the implementation of the research presented at the [**WISE2024**](https://wise2024-qatar.com), under the title: **"Periphoscape: Enhance Wikipedia Browsing by Presenting Diverse Aspects of Topics"**. For a detailed explanation of the research, please refer to [the conference proceedings](https://link.springer.com/chapter/10.1007/978-981-96-0579-8_25).
## Data

This implementation utilizes data from [Wikipedia dumps](https://dumps.wikimedia.org) to understand the relationships between Wikipedia articles and categories. Additionally, article content for each section (paragraph) is retrieved from [Wikimedia Enterprise HTML Dump](https://dumps.wikimedia.org/other/enterprise_html/).

## Demo

The repository includes **Demo.ipynb**, which is provided as a demo of the system. To ensure the demo runs properly, a minimal dataset has been extracted from a set of Wikipedia dumps and included in the repository. Specifically, these are the files from which the dataset has been extracted:
- `jawiki-20240220-page.sql.gz`
- `jawiki-20240220-pagelinks.sql.gz`
- `jawiki-20240220-category.sql.gz`
- `jawiki-20240220-categorylinks.sql.gz`
- `jawiki-NS0-20240220-ENTERPRISE-HTML.json.tar.gz`

## Tested Environment

This repository is confirmed to work with the following environment:
- **Python**: 3.8.13
- **Libraries**:
  - `numpy==1.22.4`
  - `matplotlib==3.5.1`
  - `networkx==3.1`
  - `pandas==1.3.5`
  - `scipy==1.10.1`

## Reference
```bibtex
@inproceedings{Sato2024wise,
   author = {Sato, Shin-ya},
   title = {Periphoscape: Enhance Wikipedia Browsing by Presenting Diverse Aspects of Topics},
   booktitle = {Web Information Systems Engineering -- WISE 2024. Lecture Notes in Computer Science},
   volume = 15436,
   pages = {352--366},
   year = 2024,
   publisher = {Springer Nature Singapore},
   doi = {10.1007/978-981-96-0579-8_25}
}
```
