# Aim
This project aims to determine if earthquakes, in the current age, are occurring more or less than frequently than they should.

# Methodology

## Overview of Methodology

Suppose there exists a ideal predictive model which is able to perfectly predict the annual number of earthquakes for years 1750-1999 using the factors described in current geological theories. This is, the model perfectly understands the relations between the factors and the annual earthquake numbers from 1750-1999. 

If the same model is unable to predict the annual earthquake numbers for years 2000-2017, this implies that the relations between the factors and annual earthquake numbers have changed. This can mean that the importance of each contributing factor has changed and/or there are more contributing factors. Regardless of the cause of the change, we can necessarily infer that the geological theories that we rely on to understand earthquakes are failing and that the current number of earthquakes cannot be explained by our current geological theories. 

## Main Software and Frameworks Used

1. **JupyterLab** was used as the development environment.
2. **pandas** and **numpy** were used for construction of datasets.
3. **matplotlib** was used for the plotting of data.
4. **pytorch** was used for the implementation of neural networks
5. **scikit-learn** was used for the implementation of traditional ML algorithms.
6. **Ray** was used for hyperparameter optimisation of neural networks
7. **scikit-optimise** was used for hyperparameter optimisation of traditional ML algorithms

## General Steps
1. Get data
2. Clean data
3. Dedicate data for years 1750 - 1999 as non test data (which is further split into training and validation data) and current data for years 2000 - 2017 as test data
4. Train different machine learning algorithms on the non test data
5. Choose the algorithm with the best performance on non test data
6. Generate predictions for years 2000 - 2017 and compare against actual values

### Step 1

#### Determination of Factors

According to academic literature and availability of data, the chosen factors which contribute to earthquakes are temperature fluctuations, construction of dams, loss of global forest cover, volcanic eruptions and material extraction. 

According to two sources, global temperature fluctuations contribute to earthquakes. 
 - https://iopscience.iop.org/article/10.1088/1755-1315/167/1/012018/pdf
 - https://link.springer.com/article/10.1007/s12517-021-09229-y
 
Due to additional pressure which can deform soil and rock, the construction of dams and reservoirs can contribute to earthquakes. - 
 - https://www.ias.ac.in/article/fulltext/reso/004/11/0004-0013. 

Similarly, so can material extraction, in the form of mining and fracking for materials, as shown by a variety of sources

 - https://www.jstor.org/stable/43432868
 - https://www.techtimes.com/articles/48828/20150426/usgs-confirms-wasterwater-fracking-causes-earthquakes-finally.htm
 - https://www.theguardian.com/world/2015/apr/24/earthquakes-fracking-drilling-us-geological-survey
 - https://www.theguardian.com/world/2015/apr/23/oil-gas-drilling-triggers-man-made-earthquakes-usgs
 
Volcanic eruptions can also cause earthquakes as evidenced in a publication. 
 - https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2018GL079060
 
A loss of forest cover implies a loss of trees and their roots, causing rapid soil erosion which in turn causes earthquakes 
 - https://www.nature.com/articles/ncomms6564

Due to the the lack of data from 1750 to recent years for global material extraction, data on global energy consumption is considered instead due to its relative availability and its correlation with global material extraction.

#### Data Sources

Global Forest Cover

1. For years 1750 to 2015, data was sourced from the supplementary material of a 2020 paper published by Land (https://doi.org/10.3390/land9050129). According to the Anthromes v2.1 classification system detailed in the paper, we deemed classes 52, 53, 54 and 61 to be forested areas. 

2. For years 2016 and 2017, data was sourced from the Forest Resources Assessment 2020 by the UN FAO. Conversion from hectares to square kilometeres was made. 

Number of Earthquakes

1. For years 1750 to 2017, data was sourced from National Centers for Environmental Information (https://www.ngdc.noaa.gov/hazel/view/hazards/earthquake/search). No constraints for event parameters were used. 

Number of Volcanic Eruptions

1. For years 1750 to 2017, data was sourced from National Centers for Environmental Information (https://www.ngdc.noaa.gov/hazel/view/hazards/volcano/event-search). No constraints for event parameters were used. 

Number of Dams Constructed

1. For years 1750-2017, data was sourced from the Global Reservoir and Dam Database v1.3 (https://globaldamwatch.org/grand/). Although other dam datasets are available (e.g. the wider covering World Reservoir Database by the International Commission on Large Dams and the GlObal geOreferenced Database of Dams), the dataset was chosen due to its wide availability and its listing of the dates of completion of the dams. While every single dam in the world may not be documented in the dataset, the dataset can still be used effectively as long as the trend of the frequency of dam construction is being represented accurately.

Global Energy Consumption

1. The goal of the study is to determine whether modern day earthquake frequencies can still be determined using traditional variables as used in the past. This necessarily requires a long running timeframe from 1750 (start of industrial revolution) to the modern age. Although the variables of global annual material extraction (e.g. mining and fracking) output or annual sizes of world material extraction industry were chosen since they had the highest probability of directly explaining earthquakes, historical data on these variables could not be found. Thus, the variable of annual world energy consumption was used because of the availability of historical data as well as how its trends mimic that of the above variables. 

2. For years 1700 and 1760, data from the UK, sourced from (https://www.sciencedirect.com/science/article/abs/pii/0301421579900491), is used to construct world data by assuming that the average UK citizen's energy habits are not dissimilar to his non-UK counterparts. (Conversion of tce/head to exajoules was done using world population data found at https://ourworldindata.org/grapher/population?time=1700..latest&country=~OWID_WRL). 

3. For years 1800-2015, the data was sourced from this paper (https://www.sciencedirect.com/science/article/abs/pii/0301421579900491). Data for missing years were interpolated. 

4. For years 1750-1800, data will be interpolated. 

5. For years 2016-2017, data from the BP Statistical Review 2021 was used (https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/pdfs/energy-economics/statistical-review/bp-stats-review-2022-full-report.pdf). 

Temperature Fluctuation

1. For years 1750-2017, data was sourced from Berkeley Earth (http://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_complete.txt). Monthly raw data was condensed into annual data for coordination with other input data. Data comes in the form of anomalies from the 1951-1980 average value. 

Additional Notes

1. For the datasets above, values were not provided for every single year from 1750-2017. Data for the missing years were constructed via the `pandas.DataFrame.interpolate()` method in the pandas library using a linear method.

2. The reason why 2017 was chosen as the final year of consideration was because of the limited availability of dam datasets. The dataset chosen was the only publicly available one which met the basic requirements of the study. However, it only contained data up till 2017. Thus, a compromise was made to reduce the final year of consideration from 2021 to 2017 as the dataset allows for the consideration of the important variable of subsurface pressure created via dam construction.

### Step 4 

This purpose of this step is the determination of the closest possible variant to the ideal predictive model mentioned above. 

The different machine learning algorithms were selected based on their suitability for regression problems and analysing time series data. The traditional algorithms chosen were Multiple Linear Regression, Support Vector Machines, Random Forest, K-Nearest Neighbors, AdaBoost and XGBoost. The scikit-learn implementations of the first five algorithms were used and . The neural networks used were Multi Layer Perceptrons and three Recurrent Neural Network variants (Elman Network, Long Short Term Memory and Gated Recurrent Units). All of them were implemented via PyTorch. All traditional and non traditional algorithms were trained on the same training data and benchmarked on the same validation data. 

Due to the difference that a change in hyperparameters can do to model accuracy, child variants of the general algorithms were generated based on hyperparameter differences. The best child variant of each algorithm was chosen based on it having the highest validation accuracy compared to its peers. This was achieved using the scikit-optimize and Tune libraries. The best child variant of each algorithm compared with its counterparts of other algorithms based on validation accuracy and the best model is thus decided. 

### Step 5 and 6

The metric used to determine the model with the highest validation accuracy is the Mean Absolute Percentage Error (MAPE). The [Pytorch implementation](https://torchmetrics.readthedocs.io/en/stable/regression/mean_absolute_percentage_error.html) was utilised. The metric allows for benchmarking across different datasets due to its definition and derivation. 

The metric used to determine if the model is able to predict new data accurately is also the MAPE. 

In a peer-reviewed article regarding the use of MAPE and its variants in social science research (written by David A. Swanson and published in Review in Economics and Finance), accuracy benchmarks for MAPE values are detailed. (Note: The article can be found at https://escholarship.org/uc/item/1f71t3x9) An MAPE value of less than 5% indicates an acceptably accurate forecast. An MAPE value that is greater than 10% but lower than 25% indicates low but acceptable accuracy. An MAPE value that is greater than 25% indicates a very low accuracy and an unacceptable forecast. As such, if the model generates an MAPE value of greater than 25%, it is possible to infer that the current frequency of earthquakes is unexpected and unexplainable by conventional factors and vice versa.  

## Limitations
1. Although attempts were made to take into account all possible factors in current geographical theories, some factors unfortunately had to be left out due to a severe lack of data. For example, frequent measurements of tectonic movement (from 1750-2017) in the form of numerical values could not be found online. 

2. As mentioned above, a large and crucial segment of the project is dedicated towards deriving a closest possible variant of the ideal predictive model. Even after choosing algorithms that are commonly relied on for time series regression and utilising hyperparameter optimisation, the resulting model may still be a far cry from the ideal model. 

## Directory Guide
| Folder Name | Purpose/Content |
| :---: | :---: |
| Raw Data | Unmodified data downloaded from sources (mentioned in Step 1) |
| Modified Data | Modified and cleaned data before timeframe standardisation |
| Modified Data (1750 to 2017) | Modified data after timeframe standardisation |
| Arrays | Timeframe standardised and cleaned data in arrays after scaling |
| Params | Parameters of algorithms with optimal hyperparameter configurations |
| Processes | Contains various Jupyter Notebooks named after their purpose |



# Results

## Neural Network Algorithms

| Name of Algorithm | Best hyperparameter configuration | Best final validation loss |
| :---: | :---: | :---: |
| MLP | {‘l1’: 8, ‘l2’: 256, ‘lr’: 0.000678082780900368, ‘batch_size’: 16, ‘hidden_dim’: 6, ‘n_layers’: 1} | 0.17653940990567207 |
| RNN | {'l1': 8, 'l2': 32, 'lr': 0.00041848007491550944, 'batch_size': 4, 'hidden_dim': 10, 'n_layers': 10} | 0.1894533668573086 |
| LSTM | {‘l1’: 128, ‘l2’: 8, ‘lr’: 0.0005815330559615102, ‘batch_size’: 16, ‘hidden_dim’: 6, ‘n_layers’: 4} | 0.17347236722707748 |
| GRU | {'l1': 8, 'l2': 32, 'lr': 0.00033825890048956196, 'batch_size': 8, 'hidden_dim': 5, 'n_layers': 6} | 0.23235508905989782

Parameters of each are stored in "Best{Name of Algorithm}.pt"

## Traditional ML Algorithms

| Name of Algorithm | Best hyperparameter configuration | Best final validation loss |
| :---: | :---: | :---: |

Parameters of each are stored in "Best{Name of Algorithm}.txt"

Based on the validation losses, it is possible to determine that a long short term memory recurrent neural network, using conventional factors, is the most suitable for predicting the annual number of earthquakes from 1750 to 2000. 

When the output of the LSTM for 2000-2017 was compared against actual values, MAPE was valued at 1.0006967782974243. Naturally, this is a high error value and suggests that the trained LSTM is unable to predict . This suggests a low causation and correlation between accepted factors of earthquakes and post 2000 earthquake occurrences. Since modern geological theories are unable to predict and explain the number of earthquakes, the current numbers of earthquakes are said to be unexplainable and unexpected.
