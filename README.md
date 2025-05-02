# Predicting Injury Severity from Automated Driving Systems (ADS) Incidents

## Introduction

Repository for Massive Data Institute with Professor Robin Dillon-Merrill Spring 2025.This study develops a random forest to predict injury severity levels (Minor, Moderate, Severe) in ADS incidents data from 1,489 NHTSA reports. Including both structured (weather, speed, motion) and unstructured data (crash narrative), our model identifies key predictive factors for each incident.

## Main Directories

- **input/**: Contains data used for the project
  - `ads.xlsx`: Filtered data for the 1,489 ads-related incidents from the reporting entities. Columns not relevant for this research have been removed
  - `SGO-2021-01_Incident_Reports_ADS.csv`: Raw data for 1,489 ads-related incidents from the reporting entities

- **output/**: Contains outputs, graphs and images of the project
  - `decision_tree_visualization.pdf`: Decision tree that shows split nodes, middle nodes of how the DT makes the decision for best split, using dtreeviz library
  - `decision_tree_visualization.html`: Decision tree that shows split nodes, middle nodes of how the DT makes the decision for best split, using dtreeviz library
  - `lda_visualization.html`: Gensim LDA visualization that show how each topic is represented in the dataset and it also shows the group of words, related to each topic
  - `output.png`: Decision tree that shows split nodes, middle nodes of how the DT makes the decision for best split
  - `shapoutput.png`: Graph showing the SHAP values for each one of the Severity of Injury classification, beehive visualization

- **documentation/**: Contains papers, data dictionaries and references to guide research structure
  - `crash thems topic modeling.pdf`: Crash Themes in Automated Vehicles: A Topic Modeling Analysis of the California Department of Motor Vehicles Automated Vehicle Crash Database
  - `Exploratory_analysis_of_injury_severity_under_diff.pdf`: Exploratory analysis of injury severity under different levels of driving automation (SAE Level 2-5) using multi-source 
  - `Predictive analytics in autonomous vehicles safety  crash outcome modeling.pdf`: Predictive analytics in autonomous vehicles safety: crash outcome modeling
  - `SGO-2021-01_Data_Element_Definitions.pdf`: Standing General Order 2021-01 Incident Report, Data Definitions
  - `using imabalanced data.pdf`: Using Random Forest to Learn Imbalanced Data
  - `project-document-spring-2025.pdf`: Documentation of the project

- **poster/**: Contains Jupyter notebooks for various experiments and analyses.
  - `mdi-scholar-Soto-research-poster.pdf`: Poster presented at MDI Spring 2025 Showcase

- **main.ipynb**: Jupyternotebook with the main code base for the project

## Data

The dataset being used comes from the United States Department of Transportation, National Highway Traffic Safety Administration (NHTSA). This dataset contains 1,489 ads-related accidents in the United States from 2021 to 2025. Please follow this link, to reference our dataset. [NTHSA Standing Order Reports and Data](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting)

## Findings

Overall accuracy 93% (96% Minor, 69% Moderate, 50% High)
- Seatbelt and Impact location (sideways) emerged as strong
predictor.
- Pre-crash speed and specific pre-crash movements were
highly predictive. Involvement of heavy trucks -> Severe

## Limitations

- Critical class imbalance towards minor injury type of incidents
- Redacted Data hinder deeper research for specific companies
- Nuanced factors (speed – > injury) need further review to understand how to interpret them

## Conclusion

- There is a need for further analysis with qualitative data analysis.
- As well as a deeper understanding of the narratives with topic modelling (casts a light on how features interact speed->injury)
- Redaction from manufacturers, can cast shadow on revealing datasets
- They only way to improve safety is through disclosing information and analyzing it