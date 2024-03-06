# Covid Case Count

### Problem Statement

The primary aim of this analysis is to highlight the key factors that contributed to confirmation of Covid-19 cases in the United States of America. We will test a number of models including: Linear Regression, Lasso, Ridge, ElasticNet, Random Forest and Personal Component Regression and focus on the model that performs the best. The goal is to create a model that can be used to predict future Covid-19 cases and cases of a future disease of similar magnitude. 

### Data

The data that we are using for this project are several datasets containing information about Covid 19, they were exported from Google BigQuery. Originally we had 5 datasets: Mobility, Open, Policy, Mask, and Symptoms. Mobility, Policy, and Open were combined into clean_covid.csv (data dictionary below). 

The mask.csv describes how often people people wore masks in different counties. The data set contains answers to a survey conducted by The New York Times. Survey participants were asked “How often do you wear a mask in public when you expect to be within six feet of another person?”. 

The symptoms.csv contains aggregrated trends of health symptoms input into the google search engine. The symptoms dataset is separated by date and county.

##### Data Dictionary for clean_covid.csv:
| COLUMN | DTYPE | DESCRIPTION | EXAMPLE COLUMNS |
|-|-|-|-|
| state | object | state the data was collected | Alabama |
| date | object | date the data was collected on | 2020-02-15 |
| retail_and_recreation_percent_change_from_baseline | float64 | mobility trends | 5.0 |
| grocery_and_pharmacy_percent_change_from_baseline | float64 | mobility trends | 2.0 |
| parks_percent_change_from_baseline | float64 | mobility trends | 39.0 |
| transit_stations_percent_change_from_baseline | float64 | mobility trends | 7.0 |
| workplaces_percent_change_from_baseline | float64 | mobility trends | 2.0 |
| residential_percent_change_from_baseline | float64 | mobility trends | -1.0 |
| aggregation_level | int64 | level of clustering | 1 |
| average_temperature_celsius | float64 | average temperature | 5.916667 |
| minimum_temperature_celsius | float64 | minimum temperature | -1.800000 |
| maximum_temperature_celsius | float64 | maximum temperature | 15.622222 |
| rainfall_mm | float64 | amount of rain in millimeters | 0.000000 |
| school_closing | float64 | Scale of 0-3, 0 - No measures 1 - recommend closing 2 - Require closing (only some levels or categories, eg just high school, or just public schools) 3 - Require closing all levels | 0.0 |
| workplace_closing | float64 | on a scale of 0-3, 0 - No measures 1 - recommend closing (or work from home) 2 - require closing (or work from home) for some sectors or categories of workers 3 - require closing (or work from home) all-but-essential workplaces (eg grocery stores, doctors) | 0.0 |
| cancel_public_events | float64 | on a scale of 0-2, 0- No measures 1 - Recommend cancelling 2 - Require cancelling | 0.0 |
| restrictions_on_gatherings | float64 | on a scale of 0-4, 0 - No restrictions 1 - Restrictions on very large gatherings (the limit is above 1000 people) 2 - Restrictions on gatherings between 100-1000 people 3 - Restrictions on gatherings between 10-100 people 4 - Restrictions on gatherings of less than 10 people | 0.0 |
| close_public_transit | float64 | on a scale of 0-2, 0 - No measures 1 - Recommend closing (or significantly reduce volume/route/means of transport available) 2 - Require closing (or prohibit most citizens from using it) | 0.0 |
| stay_at_home_requirements | float64 | scale record of orders to “shelter-in- place” and otherwise confine to home. | 0.0 |
| restrictions_on_internal_movement | float64 | on a scale of 0-2, 0 - No measures 1 - Recommend closing (or significantly reduce volume/route/means of transport) 2 - Require closing (or prohibit most people from using it) | 0.0 |
| international_travel_controls | float64 | on a scale of 0-4, 0 - No measures 1 - Screening 2 - Quarantine arrivals from high-risk regions 3 - Ban on high-risk regions 4 - Total border closure | 2.0 |
| income_support | float64 | scale record if the government is covering the salaries or providing direct cash payments, universal basic income, or similar, of people who lose their jobs or cannot work | 0.0 |
| debt_contract_relief | float64 | record if govt. is freezing financial obligations | 0.0 |
| fiscal_measures | float64 | stimulus in USD | 0.0 |
| international_support | float64 | covid 18 aid for other countries in USD | 0.0 |
| public_information_campaigns | float64 | on a scale of 0-2, 0 -No COVID-19 public information campaign 1 - public officials urging caution about COVID-19 2 - coordinated public information campaign (e.g. across traditional and social media) | 0.0 |
| testing_policy | float64 | on a scale of 0-3, 0 – No testing policy 1 – Only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers, admitted to hospital, came into contact with a known case, returned from overseas) 2 – testing of anyone showing COVID-19 symptoms 3 – open public testing (eg “drive through” testing available to asymptomatic people) | 1.0 |
| contact_tracing | float64 | on a scale of 0-2, 0 - No contact tracing 1 - Limited contact tracing - not done for all cases 2 - Comprehensive contact tracing - done for all cases | 0.0 |
| emergency_healthcare_investment | float64 | emergency spending for covid in USD | 0.0 |
| vaccine_investment | float64 | public spending on a vaccine in USD | 0.0 |
| confirmed_cases | float64 | number of confirmed cases | 0.0 |
| deaths | float64 | number of deaths | 0.0 |
| stringency_index | float64 | on a scale of 1-100, how well the government responded to covid. based on 9 indicators that they could have acted upon and to what degree| 5.56 |

##### Data Description for Symptoms:
Columns are health symptoms, signs and conditions. Rows contain data points that reflecting the volume of Google searches. The data is broken down by county and date. 

##### Data Description for Mask Dataset:
countyfp: The county FIPS code.

never: The estimated share of people in this county who would say never in response to the question 

rarely: The estimated share of people in this county who would say rarely

sometimes: The estimated share of people in this county who would say sometimes

frequently: The estimated share of people in this county who would say frequently

always: The estimated share of people in this county who would say always

### EDA
<img src="https://github.com/smohan04/Covid-Case-Count/tree/main/images/symptom_image.png" width="500" height="500">

* Searches for infection, common cold and fever occurred much more this year 2020 then last year

* Many of the top correlated terms refer to behaviorial symptoms instead of more concrete symptoms. Terms like dysphoria, confusion, depression, anxiety.

   ![](https://github.com/smohan04/Covid-Case-Count/tree/main/images/percentages.png)
   
   ![](https://github.com/smohan04/Covid-Case-Count/tree/main/images/Covid-19_Case_Count_Analysis.jpg)

### Results

##### Models and Scores:
###### Open Data
| Models | Training R2 Score | Testing R2 Score |
|-|-|-|
| ENET | 0.6425 | 0.615183 |
| Linear_Regression | 0.6819 | 0.654500 |
| Ridge | 0.6812 | 0.654200 |
| Lasso | 0.6817 | 0.654500 |
| Random_Forest Regression | 0.9979 | 0.988600 |
| Principle Component Regression | 0.9917 | 0.918800 |

###### Symptoms
|       Model       | Training Score | Test Score |
|:-----------------:|----------------|------------|
| Linear Regression | 0.733689       | 0.718972   |
| Ridge             | 0.733673       | 0.719133   |
| RidgeCV           | 0.733269       | 0.719450   |
| Lasso             | 0.733689       | 0.718990   |
| Random Forest     | 0.994890       | 0.969984   |

###### Masks County
|       Model       | Training Score | Test Score |
|:-----------------:|----------------|------------|
| Linear Regression | 0.648969       | -3.535922  |
| Ridge             | 0.922133       | 0.344883   |
| RidgeCV           | 0.911987       | 0.333114   |
| Lasso             | 0.401339       | 0.449235   |
| Random Forest     | 0.965044       |  0.776316  |
| Extra Tree        | 1.0            | 0.785858   |
  
### Conclusions
In conclusion, we created a multiple regression models (Linear Regression, Lasso Regression, Ridge, Elastic Net, Principle Component Regression, Extra Tree). The best performance for all our data sets was through Random Forest Regression modeling. 

The R2 score for Open Data testing data was around .989, therefore accounting for 98.9% of the variability in testing data. The Root Mean Square Error of the model was right at 5,649. Therefore, the model can accurately predict confirmed cases by state within 5,649 cases, important to note when looking at future predicted cases based off of this model.

For the symptoms dataset, we found that compared to last year, the top 5 google search symptoms were infection, common cold, fever, acne, and xeroderma. Behavioral symptoms were the most highly correlated with covid case count (ex. dysphoria, anxiety, depression). We were able to create a random forest model just on google symptom search date with a test score R2 of .969.  

In the mask data, we could not find a strong correlation between mask usage and covid case count. All the models overfit the training data and produced poor testing scores. 

