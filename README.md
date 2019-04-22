# machine-learning-gazetteer-matching

This repo features code, annotated data, and results for the IJGIS paper _[Machine learning for cross-gazetteer matching of natural features](https://www.tandfonline.com/doi/full/10.1080/13658816.2019.1599123)_.

### Notebooks
Jupyter notebooks are in the top-level of this repo, numbered according to the order in which they should be run, and organized into 3 numbered subsets:
- 0_ : (00, 01, 02): preparation, preprocessing
- 1_ : (10, 11, 12, 13, 14): rule-based matching
- 2_ : (20, 21): machine learning based matching using random forests

Note these notebooks rely heavily on code in the gazmatch folder.

### Data
In _/data/_, we share our annotated data, _annotated_sample.csv_ as well as some serialized files, including _test_set_ids.pkl_ for the feature-type-balanced test set used in a subset of experiments. The latest **GeoNames** and **SwissNames3D** data can be obtained online:
- GeoNames daily dumps: [http://download.geonames.org/export/dump/] then choose CH.zip for the Switzerland data
- SwissNames3D latest version: [https://shop.swisstopo.admin.ch/en/products/landscape/names3D]

Note these datasets will not be identical to the ones used in this paper, which were downloaded in 2017. Data preparation involving the raw datasets is described and performed in the preparation notebooks.

### Results
The _/results/_ folder contains tsv files used to plot the results in the paper. The _/html_exports/_ contains html exports of all the notebooks for easy viewing in a browser.
