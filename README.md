# DeGraphCS


# Project Overview
This project provides a collection of datasets and source code which are used in our DeGraphCS model. The content of project is as follows:

1. Appendix
 
2. Baseline methods

3. dataset  

4. IR2graph

5. src

6. user study

## Raw Datasets
To help people to reproduce our work, we provide raw datasets which are consist of **C code snippet**, corresponding **code comment** and **generated IR**.

The raw datasets can be accessed in [Google Drive](https://drive.google.com/file/d/1PZ9TAfsrSlXLDpOCp6-0aZQxrzlP4kBA/view?usp=sharing)

## Dataset
To feed into our model, we first generate Variable-based Flow Graph of 41152 methods and extract corresponding comments. Then we split the datasets into 39152 training set and 2000 test set. All of the data are puted in `dataset/` directory. 

## Baseline Methods
We have reproduced other code search works which are putted in `Baseline methods/` directory.

## IR2graph
We provide graph construction code to help users to generate Variable-based Flow Graph which are puted in `IR2graph/` directory.

## Src
We provide DeGraphCS model code are listed in `src/` directory.

## User Study
We make a user study to evaluate our model. 

50 queries of the user study are listed in the `user study/queries.txt`. And according to four models (UNIF, MMAN, DeepCS and DeGraphCS), we obtain corresponding searching result which are listed in  `user study/` directory.

# Running Our Model
## Generate Datasets and Build Dictionary
Run the command to split comments datasets into training set and test set, and build dictionary
```
python src/util_desc.py
```
Run the command to split Variable-based Flow Graph datasets into training set and test set, and build dictionary
```
python src/util_ir.py
```
## Train the DeGraphCS Model
```
python src/train.py
```
## Test the DeGraphCS Model
```
python src/test.py
```
