{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3febcd6a",
   "metadata": {},
   "source": [
    "## Context:\n",
    "The World Happiness Report is a landmark survey of the state of global happiness. \n",
    "The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9f1e480",
   "metadata": {},
   "source": [
    "## Content\n",
    "the happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale.\n",
    "The columns following the happiness score estimate the extent to which each of six factors – GDP per capita, social support, life expectancy, freedom, Perceptions of corruption, and generosity – contribute to making life evaluations higher in each country "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6079aedf",
   "metadata": {},
   "source": [
    "## Data Science question: WHAT MAKES A COUNTRY HAPPY?\n",
    "\n",
    "Guide:\n",
    "- How does each  srtong happiness factor correlate with one another in 2021?\n",
    "- What is mean of corruption in all countries in Regional indicator in 2021? \n",
    "- What the happiest and saddiest countries (top 10 and bottom 10 in 2021)?\n",
    "- How is the Healthy life expectancy factor in the happiest and saddiest counries in 2021?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "592c26f6",
   "metadata": {},
   "source": [
    "## Algoithms Performed below"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae44a734",
   "metadata": {},
   "source": [
    "### Regression Algorithm\n",
    "1)KNN Regressor 2)Linear Regressor 3)Decision Tree Regressor 4)Random Forest Regressor 5)Support Vector Regressor 6)XG Boost Regressor"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2f4b2219",
   "metadata": {},
   "source": [
    "## merge two datasets"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c2c2be3",
   "metadata": {},
   "source": [
    "### Data source: https://www.kaggle.com\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13c907a8",
   "metadata": {},
   "source": [
    "### Import Libraries \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8e755fc9",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: plotly in /opt/anaconda3/lib/python3.9/site-packages (5.5.0)\r\n",
      "Requirement already satisfied: six in /opt/anaconda3/lib/python3.9/site-packages (from plotly) (1.16.0)\r\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /opt/anaconda3/lib/python3.9/site-packages (from plotly) (8.0.1)\r\n"
     ]
    }
   ],
   "source": [
    "!pip install plotly \n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import statsmodels.api as sm\n",
    "\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import r2_score\n",
    "import plotly.express as px\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63c29826",
   "metadata": {},
   "source": [
    "### read  and cleaning data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ad3b9113",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import two Datasets\n",
    "data2021=pd.read_csv('world-happiness-report-2021.csv')\n",
    "data=pd.read_csv('world-happiness-report.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4d95fd53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>Regional indicator</th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Standard error of ladder score</th>\n",
       "      <th>upperwhisker</th>\n",
       "      <th>lowerwhisker</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "      <th>Ladder score in Dystopia</th>\n",
       "      <th>Explained by: Log GDP per capita</th>\n",
       "      <th>Explained by: Social support</th>\n",
       "      <th>Explained by: Healthy life expectancy</th>\n",
       "      <th>Explained by: Freedom to make life choices</th>\n",
       "      <th>Explained by: Generosity</th>\n",
       "      <th>Explained by: Perceptions of corruption</th>\n",
       "      <th>Dystopia + residual</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Finland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.842</td>\n",
       "      <td>0.032</td>\n",
       "      <td>7.904</td>\n",
       "      <td>7.780</td>\n",
       "      <td>10.775</td>\n",
       "      <td>0.954</td>\n",
       "      <td>72.0</td>\n",
       "      <td>0.949</td>\n",
       "      <td>-0.098</td>\n",
       "      <td>0.186</td>\n",
       "      <td>2.43</td>\n",
       "      <td>1.446</td>\n",
       "      <td>1.106</td>\n",
       "      <td>0.741</td>\n",
       "      <td>0.691</td>\n",
       "      <td>0.124</td>\n",
       "      <td>0.481</td>\n",
       "      <td>3.253</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Denmark</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.620</td>\n",
       "      <td>0.035</td>\n",
       "      <td>7.687</td>\n",
       "      <td>7.552</td>\n",
       "      <td>10.933</td>\n",
       "      <td>0.954</td>\n",
       "      <td>72.7</td>\n",
       "      <td>0.946</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.179</td>\n",
       "      <td>2.43</td>\n",
       "      <td>1.502</td>\n",
       "      <td>1.108</td>\n",
       "      <td>0.763</td>\n",
       "      <td>0.686</td>\n",
       "      <td>0.208</td>\n",
       "      <td>0.485</td>\n",
       "      <td>2.868</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Switzerland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.571</td>\n",
       "      <td>0.036</td>\n",
       "      <td>7.643</td>\n",
       "      <td>7.500</td>\n",
       "      <td>11.117</td>\n",
       "      <td>0.942</td>\n",
       "      <td>74.4</td>\n",
       "      <td>0.919</td>\n",
       "      <td>0.025</td>\n",
       "      <td>0.292</td>\n",
       "      <td>2.43</td>\n",
       "      <td>1.566</td>\n",
       "      <td>1.079</td>\n",
       "      <td>0.816</td>\n",
       "      <td>0.653</td>\n",
       "      <td>0.204</td>\n",
       "      <td>0.413</td>\n",
       "      <td>2.839</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Iceland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.554</td>\n",
       "      <td>0.059</td>\n",
       "      <td>7.670</td>\n",
       "      <td>7.438</td>\n",
       "      <td>10.878</td>\n",
       "      <td>0.983</td>\n",
       "      <td>73.0</td>\n",
       "      <td>0.955</td>\n",
       "      <td>0.160</td>\n",
       "      <td>0.673</td>\n",
       "      <td>2.43</td>\n",
       "      <td>1.482</td>\n",
       "      <td>1.172</td>\n",
       "      <td>0.772</td>\n",
       "      <td>0.698</td>\n",
       "      <td>0.293</td>\n",
       "      <td>0.170</td>\n",
       "      <td>2.967</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Netherlands</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.464</td>\n",
       "      <td>0.027</td>\n",
       "      <td>7.518</td>\n",
       "      <td>7.410</td>\n",
       "      <td>10.932</td>\n",
       "      <td>0.942</td>\n",
       "      <td>72.4</td>\n",
       "      <td>0.913</td>\n",
       "      <td>0.175</td>\n",
       "      <td>0.338</td>\n",
       "      <td>2.43</td>\n",
       "      <td>1.501</td>\n",
       "      <td>1.079</td>\n",
       "      <td>0.753</td>\n",
       "      <td>0.647</td>\n",
       "      <td>0.302</td>\n",
       "      <td>0.384</td>\n",
       "      <td>2.798</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Country name Regional indicator  Ladder score  \\\n",
       "0      Finland     Western Europe         7.842   \n",
       "1      Denmark     Western Europe         7.620   \n",
       "2  Switzerland     Western Europe         7.571   \n",
       "3      Iceland     Western Europe         7.554   \n",
       "4  Netherlands     Western Europe         7.464   \n",
       "\n",
       "   Standard error of ladder score  upperwhisker  lowerwhisker  \\\n",
       "0                           0.032         7.904         7.780   \n",
       "1                           0.035         7.687         7.552   \n",
       "2                           0.036         7.643         7.500   \n",
       "3                           0.059         7.670         7.438   \n",
       "4                           0.027         7.518         7.410   \n",
       "\n",
       "   Logged GDP per capita  Social support  Healthy life expectancy  \\\n",
       "0                 10.775           0.954                     72.0   \n",
       "1                 10.933           0.954                     72.7   \n",
       "2                 11.117           0.942                     74.4   \n",
       "3                 10.878           0.983                     73.0   \n",
       "4                 10.932           0.942                     72.4   \n",
       "\n",
       "   Freedom to make life choices  Generosity  Perceptions of corruption  \\\n",
       "0                         0.949      -0.098                      0.186   \n",
       "1                         0.946       0.030                      0.179   \n",
       "2                         0.919       0.025                      0.292   \n",
       "3                         0.955       0.160                      0.673   \n",
       "4                         0.913       0.175                      0.338   \n",
       "\n",
       "   Ladder score in Dystopia  Explained by: Log GDP per capita  \\\n",
       "0                      2.43                             1.446   \n",
       "1                      2.43                             1.502   \n",
       "2                      2.43                             1.566   \n",
       "3                      2.43                             1.482   \n",
       "4                      2.43                             1.501   \n",
       "\n",
       "   Explained by: Social support  Explained by: Healthy life expectancy  \\\n",
       "0                         1.106                                  0.741   \n",
       "1                         1.108                                  0.763   \n",
       "2                         1.079                                  0.816   \n",
       "3                         1.172                                  0.772   \n",
       "4                         1.079                                  0.753   \n",
       "\n",
       "   Explained by: Freedom to make life choices  Explained by: Generosity  \\\n",
       "0                                       0.691                     0.124   \n",
       "1                                       0.686                     0.208   \n",
       "2                                       0.653                     0.204   \n",
       "3                                       0.698                     0.293   \n",
       "4                                       0.647                     0.302   \n",
       "\n",
       "   Explained by: Perceptions of corruption  Dystopia + residual  \n",
       "0                                    0.481                3.253  \n",
       "1                                    0.485                2.868  \n",
       "2                                    0.413                2.839  \n",
       "3                                    0.170                2.967  \n",
       "4                                    0.384                2.798  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the overview of the data2021\n",
    "data2021.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d3b9bf53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Standard error of ladder score</th>\n",
       "      <th>upperwhisker</th>\n",
       "      <th>lowerwhisker</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "      <th>Ladder score in Dystopia</th>\n",
       "      <th>Explained by: Log GDP per capita</th>\n",
       "      <th>Explained by: Social support</th>\n",
       "      <th>Explained by: Healthy life expectancy</th>\n",
       "      <th>Explained by: Freedom to make life choices</th>\n",
       "      <th>Explained by: Generosity</th>\n",
       "      <th>Explained by: Perceptions of corruption</th>\n",
       "      <th>Dystopia + residual</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>1.490000e+02</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "      <td>149.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.532839</td>\n",
       "      <td>0.058752</td>\n",
       "      <td>5.648007</td>\n",
       "      <td>5.417631</td>\n",
       "      <td>9.432208</td>\n",
       "      <td>0.814745</td>\n",
       "      <td>64.992799</td>\n",
       "      <td>0.791597</td>\n",
       "      <td>-0.015134</td>\n",
       "      <td>0.727450</td>\n",
       "      <td>2.430000e+00</td>\n",
       "      <td>0.977161</td>\n",
       "      <td>0.793315</td>\n",
       "      <td>0.520161</td>\n",
       "      <td>0.498711</td>\n",
       "      <td>0.178047</td>\n",
       "      <td>0.135141</td>\n",
       "      <td>2.430329</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.073924</td>\n",
       "      <td>0.022001</td>\n",
       "      <td>1.054330</td>\n",
       "      <td>1.094879</td>\n",
       "      <td>1.158601</td>\n",
       "      <td>0.114889</td>\n",
       "      <td>6.762043</td>\n",
       "      <td>0.113332</td>\n",
       "      <td>0.150657</td>\n",
       "      <td>0.179226</td>\n",
       "      <td>5.347044e-15</td>\n",
       "      <td>0.404740</td>\n",
       "      <td>0.258871</td>\n",
       "      <td>0.213019</td>\n",
       "      <td>0.137888</td>\n",
       "      <td>0.098270</td>\n",
       "      <td>0.114361</td>\n",
       "      <td>0.537645</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2.523000</td>\n",
       "      <td>0.026000</td>\n",
       "      <td>2.596000</td>\n",
       "      <td>2.449000</td>\n",
       "      <td>6.635000</td>\n",
       "      <td>0.463000</td>\n",
       "      <td>48.478000</td>\n",
       "      <td>0.382000</td>\n",
       "      <td>-0.288000</td>\n",
       "      <td>0.082000</td>\n",
       "      <td>2.430000e+00</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.648000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>4.852000</td>\n",
       "      <td>0.043000</td>\n",
       "      <td>4.991000</td>\n",
       "      <td>4.706000</td>\n",
       "      <td>8.541000</td>\n",
       "      <td>0.750000</td>\n",
       "      <td>59.802000</td>\n",
       "      <td>0.718000</td>\n",
       "      <td>-0.126000</td>\n",
       "      <td>0.667000</td>\n",
       "      <td>2.430000e+00</td>\n",
       "      <td>0.666000</td>\n",
       "      <td>0.647000</td>\n",
       "      <td>0.357000</td>\n",
       "      <td>0.409000</td>\n",
       "      <td>0.105000</td>\n",
       "      <td>0.060000</td>\n",
       "      <td>2.138000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.534000</td>\n",
       "      <td>0.054000</td>\n",
       "      <td>5.625000</td>\n",
       "      <td>5.413000</td>\n",
       "      <td>9.569000</td>\n",
       "      <td>0.832000</td>\n",
       "      <td>66.603000</td>\n",
       "      <td>0.804000</td>\n",
       "      <td>-0.036000</td>\n",
       "      <td>0.781000</td>\n",
       "      <td>2.430000e+00</td>\n",
       "      <td>1.025000</td>\n",
       "      <td>0.832000</td>\n",
       "      <td>0.571000</td>\n",
       "      <td>0.514000</td>\n",
       "      <td>0.164000</td>\n",
       "      <td>0.101000</td>\n",
       "      <td>2.509000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.255000</td>\n",
       "      <td>0.070000</td>\n",
       "      <td>6.344000</td>\n",
       "      <td>6.128000</td>\n",
       "      <td>10.421000</td>\n",
       "      <td>0.905000</td>\n",
       "      <td>69.600000</td>\n",
       "      <td>0.877000</td>\n",
       "      <td>0.079000</td>\n",
       "      <td>0.845000</td>\n",
       "      <td>2.430000e+00</td>\n",
       "      <td>1.323000</td>\n",
       "      <td>0.996000</td>\n",
       "      <td>0.665000</td>\n",
       "      <td>0.603000</td>\n",
       "      <td>0.239000</td>\n",
       "      <td>0.174000</td>\n",
       "      <td>2.794000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.842000</td>\n",
       "      <td>0.173000</td>\n",
       "      <td>7.904000</td>\n",
       "      <td>7.780000</td>\n",
       "      <td>11.647000</td>\n",
       "      <td>0.983000</td>\n",
       "      <td>76.953000</td>\n",
       "      <td>0.970000</td>\n",
       "      <td>0.542000</td>\n",
       "      <td>0.939000</td>\n",
       "      <td>2.430000e+00</td>\n",
       "      <td>1.751000</td>\n",
       "      <td>1.172000</td>\n",
       "      <td>0.897000</td>\n",
       "      <td>0.716000</td>\n",
       "      <td>0.541000</td>\n",
       "      <td>0.547000</td>\n",
       "      <td>3.482000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Ladder score  Standard error of ladder score  upperwhisker  \\\n",
       "count    149.000000                      149.000000    149.000000   \n",
       "mean       5.532839                        0.058752      5.648007   \n",
       "std        1.073924                        0.022001      1.054330   \n",
       "min        2.523000                        0.026000      2.596000   \n",
       "25%        4.852000                        0.043000      4.991000   \n",
       "50%        5.534000                        0.054000      5.625000   \n",
       "75%        6.255000                        0.070000      6.344000   \n",
       "max        7.842000                        0.173000      7.904000   \n",
       "\n",
       "       lowerwhisker  Logged GDP per capita  Social support  \\\n",
       "count    149.000000             149.000000      149.000000   \n",
       "mean       5.417631               9.432208        0.814745   \n",
       "std        1.094879               1.158601        0.114889   \n",
       "min        2.449000               6.635000        0.463000   \n",
       "25%        4.706000               8.541000        0.750000   \n",
       "50%        5.413000               9.569000        0.832000   \n",
       "75%        6.128000              10.421000        0.905000   \n",
       "max        7.780000              11.647000        0.983000   \n",
       "\n",
       "       Healthy life expectancy  Freedom to make life choices  Generosity  \\\n",
       "count               149.000000                    149.000000  149.000000   \n",
       "mean                 64.992799                      0.791597   -0.015134   \n",
       "std                   6.762043                      0.113332    0.150657   \n",
       "min                  48.478000                      0.382000   -0.288000   \n",
       "25%                  59.802000                      0.718000   -0.126000   \n",
       "50%                  66.603000                      0.804000   -0.036000   \n",
       "75%                  69.600000                      0.877000    0.079000   \n",
       "max                  76.953000                      0.970000    0.542000   \n",
       "\n",
       "       Perceptions of corruption  Ladder score in Dystopia  \\\n",
       "count                 149.000000              1.490000e+02   \n",
       "mean                    0.727450              2.430000e+00   \n",
       "std                     0.179226              5.347044e-15   \n",
       "min                     0.082000              2.430000e+00   \n",
       "25%                     0.667000              2.430000e+00   \n",
       "50%                     0.781000              2.430000e+00   \n",
       "75%                     0.845000              2.430000e+00   \n",
       "max                     0.939000              2.430000e+00   \n",
       "\n",
       "       Explained by: Log GDP per capita  Explained by: Social support  \\\n",
       "count                        149.000000                    149.000000   \n",
       "mean                           0.977161                      0.793315   \n",
       "std                            0.404740                      0.258871   \n",
       "min                            0.000000                      0.000000   \n",
       "25%                            0.666000                      0.647000   \n",
       "50%                            1.025000                      0.832000   \n",
       "75%                            1.323000                      0.996000   \n",
       "max                            1.751000                      1.172000   \n",
       "\n",
       "       Explained by: Healthy life expectancy  \\\n",
       "count                             149.000000   \n",
       "mean                                0.520161   \n",
       "std                                 0.213019   \n",
       "min                                 0.000000   \n",
       "25%                                 0.357000   \n",
       "50%                                 0.571000   \n",
       "75%                                 0.665000   \n",
       "max                                 0.897000   \n",
       "\n",
       "       Explained by: Freedom to make life choices  Explained by: Generosity  \\\n",
       "count                                  149.000000                149.000000   \n",
       "mean                                     0.498711                  0.178047   \n",
       "std                                      0.137888                  0.098270   \n",
       "min                                      0.000000                  0.000000   \n",
       "25%                                      0.409000                  0.105000   \n",
       "50%                                      0.514000                  0.164000   \n",
       "75%                                      0.603000                  0.239000   \n",
       "max                                      0.716000                  0.541000   \n",
       "\n",
       "       Explained by: Perceptions of corruption  Dystopia + residual  \n",
       "count                               149.000000           149.000000  \n",
       "mean                                  0.135141             2.430329  \n",
       "std                                   0.114361             0.537645  \n",
       "min                                   0.000000             0.648000  \n",
       "25%                                   0.060000             2.138000  \n",
       "50%                                   0.101000             2.509000  \n",
       "75%                                   0.174000             2.794000  \n",
       "max                                   0.547000             3.482000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#describe of data2021\n",
    "data2021.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ca382c30",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 149 entries, 0 to 148\n",
      "Data columns (total 20 columns):\n",
      " #   Column                                      Non-Null Count  Dtype  \n",
      "---  ------                                      --------------  -----  \n",
      " 0   Country name                                149 non-null    object \n",
      " 1   Regional indicator                          149 non-null    object \n",
      " 2   Ladder score                                149 non-null    float64\n",
      " 3   Standard error of ladder score              149 non-null    float64\n",
      " 4   upperwhisker                                149 non-null    float64\n",
      " 5   lowerwhisker                                149 non-null    float64\n",
      " 6   Logged GDP per capita                       149 non-null    float64\n",
      " 7   Social support                              149 non-null    float64\n",
      " 8   Healthy life expectancy                     149 non-null    float64\n",
      " 9   Freedom to make life choices                149 non-null    float64\n",
      " 10  Generosity                                  149 non-null    float64\n",
      " 11  Perceptions of corruption                   149 non-null    float64\n",
      " 12  Ladder score in Dystopia                    149 non-null    float64\n",
      " 13  Explained by: Log GDP per capita            149 non-null    float64\n",
      " 14  Explained by: Social support                149 non-null    float64\n",
      " 15  Explained by: Healthy life expectancy       149 non-null    float64\n",
      " 16  Explained by: Freedom to make life choices  149 non-null    float64\n",
      " 17  Explained by: Generosity                    149 non-null    float64\n",
      " 18  Explained by: Perceptions of corruption     149 non-null    float64\n",
      " 19  Dystopia + residual                         149 non-null    float64\n",
      "dtypes: float64(18), object(2)\n",
      "memory usage: 23.4+ KB\n"
     ]
    }
   ],
   "source": [
    "# Information about the Variables\n",
    "data2021.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "90580ff2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Country name                                  0\n",
       "Regional indicator                            0\n",
       "Ladder score                                  0\n",
       "Standard error of ladder score                0\n",
       "upperwhisker                                  0\n",
       "lowerwhisker                                  0\n",
       "Logged GDP per capita                         0\n",
       "Social support                                0\n",
       "Healthy life expectancy                       0\n",
       "Freedom to make life choices                  0\n",
       "Generosity                                    0\n",
       "Perceptions of corruption                     0\n",
       "Ladder score in Dystopia                      0\n",
       "Explained by: Log GDP per capita              0\n",
       "Explained by: Social support                  0\n",
       "Explained by: Healthy life expectancy         0\n",
       "Explained by: Freedom to make life choices    0\n",
       "Explained by: Generosity                      0\n",
       "Explained by: Perceptions of corruption       0\n",
       "Dystopia + residual                           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#missing data\n",
    "data2021.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e5a6854a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Country name', 'Regional indicator', 'Ladder score',\n",
       "       'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',\n",
       "       'Logged GDP per capita', 'Social support', 'Healthy life expectancy',\n",
       "       'Freedom to make life choices', 'Generosity',\n",
       "       'Perceptions of corruption', 'Ladder score in Dystopia',\n",
       "       'Explained by: Log GDP per capita', 'Explained by: Social support',\n",
       "       'Explained by: Healthy life expectancy',\n",
       "       'Explained by: Freedom to make life choices',\n",
       "       'Explained by: Generosity', 'Explained by: Perceptions of corruption',\n",
       "       'Dystopia + residual'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#columns in data2021\n",
    "data2021.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fa6770be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(149, 20)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#size of data2021\n",
    "data2021.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "978c467c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop unnecessary columns\n",
    "data2021 = data2021.drop(['Standard error of ladder score', 'upperwhisker', 'lowerwhisker',\n",
    "       'Explained by: Log GDP per capita', 'Explained by: Social support',\n",
    "       'Explained by: Healthy life expectancy',\n",
    "       'Explained by: Freedom to make life choices',\n",
    "       'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Ladder score in Dystopia',\n",
    "                          'Dystopia + residual'], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0913e679",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>Regional indicator</th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Finland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.842</td>\n",
       "      <td>10.775</td>\n",
       "      <td>0.954</td>\n",
       "      <td>72.0</td>\n",
       "      <td>0.949</td>\n",
       "      <td>-0.098</td>\n",
       "      <td>0.186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Denmark</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.620</td>\n",
       "      <td>10.933</td>\n",
       "      <td>0.954</td>\n",
       "      <td>72.7</td>\n",
       "      <td>0.946</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.179</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Switzerland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.571</td>\n",
       "      <td>11.117</td>\n",
       "      <td>0.942</td>\n",
       "      <td>74.4</td>\n",
       "      <td>0.919</td>\n",
       "      <td>0.025</td>\n",
       "      <td>0.292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Iceland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.554</td>\n",
       "      <td>10.878</td>\n",
       "      <td>0.983</td>\n",
       "      <td>73.0</td>\n",
       "      <td>0.955</td>\n",
       "      <td>0.160</td>\n",
       "      <td>0.673</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Netherlands</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.464</td>\n",
       "      <td>10.932</td>\n",
       "      <td>0.942</td>\n",
       "      <td>72.4</td>\n",
       "      <td>0.913</td>\n",
       "      <td>0.175</td>\n",
       "      <td>0.338</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Country name Regional indicator  Ladder score  Logged GDP per capita  \\\n",
       "0      Finland     Western Europe         7.842                 10.775   \n",
       "1      Denmark     Western Europe         7.620                 10.933   \n",
       "2  Switzerland     Western Europe         7.571                 11.117   \n",
       "3      Iceland     Western Europe         7.554                 10.878   \n",
       "4  Netherlands     Western Europe         7.464                 10.932   \n",
       "\n",
       "   Social support  Healthy life expectancy  Freedom to make life choices  \\\n",
       "0           0.954                     72.0                         0.949   \n",
       "1           0.954                     72.7                         0.946   \n",
       "2           0.942                     74.4                         0.919   \n",
       "3           0.983                     73.0                         0.955   \n",
       "4           0.942                     72.4                         0.913   \n",
       "\n",
       "   Generosity  Perceptions of corruption  \n",
       "0      -0.098                      0.186  \n",
       "1       0.030                      0.179  \n",
       "2       0.025                      0.292  \n",
       "3       0.160                      0.673  \n",
       "4       0.175                      0.338  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data2021.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d2d3a52e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>year</th>\n",
       "      <th>Life Ladder</th>\n",
       "      <th>Log GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy at birth</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "      <th>Positive affect</th>\n",
       "      <th>Negative affect</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2008</td>\n",
       "      <td>3.724</td>\n",
       "      <td>7.370</td>\n",
       "      <td>0.451</td>\n",
       "      <td>50.80</td>\n",
       "      <td>0.718</td>\n",
       "      <td>0.168</td>\n",
       "      <td>0.882</td>\n",
       "      <td>0.518</td>\n",
       "      <td>0.258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2009</td>\n",
       "      <td>4.402</td>\n",
       "      <td>7.540</td>\n",
       "      <td>0.552</td>\n",
       "      <td>51.20</td>\n",
       "      <td>0.679</td>\n",
       "      <td>0.190</td>\n",
       "      <td>0.850</td>\n",
       "      <td>0.584</td>\n",
       "      <td>0.237</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2010</td>\n",
       "      <td>4.758</td>\n",
       "      <td>7.647</td>\n",
       "      <td>0.539</td>\n",
       "      <td>51.60</td>\n",
       "      <td>0.600</td>\n",
       "      <td>0.121</td>\n",
       "      <td>0.707</td>\n",
       "      <td>0.618</td>\n",
       "      <td>0.275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2011</td>\n",
       "      <td>3.832</td>\n",
       "      <td>7.620</td>\n",
       "      <td>0.521</td>\n",
       "      <td>51.92</td>\n",
       "      <td>0.496</td>\n",
       "      <td>0.162</td>\n",
       "      <td>0.731</td>\n",
       "      <td>0.611</td>\n",
       "      <td>0.267</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2012</td>\n",
       "      <td>3.783</td>\n",
       "      <td>7.705</td>\n",
       "      <td>0.521</td>\n",
       "      <td>52.24</td>\n",
       "      <td>0.531</td>\n",
       "      <td>0.236</td>\n",
       "      <td>0.776</td>\n",
       "      <td>0.710</td>\n",
       "      <td>0.268</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Country name  year  Life Ladder  Log GDP per capita  Social support  \\\n",
       "0  Afghanistan  2008        3.724               7.370           0.451   \n",
       "1  Afghanistan  2009        4.402               7.540           0.552   \n",
       "2  Afghanistan  2010        4.758               7.647           0.539   \n",
       "3  Afghanistan  2011        3.832               7.620           0.521   \n",
       "4  Afghanistan  2012        3.783               7.705           0.521   \n",
       "\n",
       "   Healthy life expectancy at birth  Freedom to make life choices  Generosity  \\\n",
       "0                             50.80                         0.718       0.168   \n",
       "1                             51.20                         0.679       0.190   \n",
       "2                             51.60                         0.600       0.121   \n",
       "3                             51.92                         0.496       0.162   \n",
       "4                             52.24                         0.531       0.236   \n",
       "\n",
       "   Perceptions of corruption  Positive affect  Negative affect  \n",
       "0                      0.882            0.518            0.258  \n",
       "1                      0.850            0.584            0.237  \n",
       "2                      0.707            0.618            0.275  \n",
       "3                      0.731            0.611            0.267  \n",
       "4                      0.776            0.710            0.268  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the overview of the data\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "93b1576f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year</th>\n",
       "      <th>Life Ladder</th>\n",
       "      <th>Log GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy at birth</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "      <th>Positive affect</th>\n",
       "      <th>Negative affect</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1949.000000</td>\n",
       "      <td>1949.000000</td>\n",
       "      <td>1913.000000</td>\n",
       "      <td>1936.000000</td>\n",
       "      <td>1894.000000</td>\n",
       "      <td>1917.000000</td>\n",
       "      <td>1860.000000</td>\n",
       "      <td>1839.000000</td>\n",
       "      <td>1927.000000</td>\n",
       "      <td>1933.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2013.216008</td>\n",
       "      <td>5.466705</td>\n",
       "      <td>9.368453</td>\n",
       "      <td>0.812552</td>\n",
       "      <td>63.359374</td>\n",
       "      <td>0.742558</td>\n",
       "      <td>0.000103</td>\n",
       "      <td>0.747125</td>\n",
       "      <td>0.710003</td>\n",
       "      <td>0.268544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>4.166828</td>\n",
       "      <td>1.115711</td>\n",
       "      <td>1.154084</td>\n",
       "      <td>0.118482</td>\n",
       "      <td>7.510245</td>\n",
       "      <td>0.142093</td>\n",
       "      <td>0.162215</td>\n",
       "      <td>0.186789</td>\n",
       "      <td>0.107100</td>\n",
       "      <td>0.085168</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2005.000000</td>\n",
       "      <td>2.375000</td>\n",
       "      <td>6.635000</td>\n",
       "      <td>0.290000</td>\n",
       "      <td>32.300000</td>\n",
       "      <td>0.258000</td>\n",
       "      <td>-0.335000</td>\n",
       "      <td>0.035000</td>\n",
       "      <td>0.322000</td>\n",
       "      <td>0.083000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2010.000000</td>\n",
       "      <td>4.640000</td>\n",
       "      <td>8.464000</td>\n",
       "      <td>0.749750</td>\n",
       "      <td>58.685000</td>\n",
       "      <td>0.647000</td>\n",
       "      <td>-0.113000</td>\n",
       "      <td>0.690000</td>\n",
       "      <td>0.625500</td>\n",
       "      <td>0.206000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>2013.000000</td>\n",
       "      <td>5.386000</td>\n",
       "      <td>9.460000</td>\n",
       "      <td>0.835500</td>\n",
       "      <td>65.200000</td>\n",
       "      <td>0.763000</td>\n",
       "      <td>-0.025500</td>\n",
       "      <td>0.802000</td>\n",
       "      <td>0.722000</td>\n",
       "      <td>0.258000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2017.000000</td>\n",
       "      <td>6.283000</td>\n",
       "      <td>10.353000</td>\n",
       "      <td>0.905000</td>\n",
       "      <td>68.590000</td>\n",
       "      <td>0.856000</td>\n",
       "      <td>0.091000</td>\n",
       "      <td>0.872000</td>\n",
       "      <td>0.799000</td>\n",
       "      <td>0.320000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2020.000000</td>\n",
       "      <td>8.019000</td>\n",
       "      <td>11.648000</td>\n",
       "      <td>0.987000</td>\n",
       "      <td>77.100000</td>\n",
       "      <td>0.985000</td>\n",
       "      <td>0.698000</td>\n",
       "      <td>0.983000</td>\n",
       "      <td>0.944000</td>\n",
       "      <td>0.705000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              year  Life Ladder  Log GDP per capita  Social support  \\\n",
       "count  1949.000000  1949.000000         1913.000000     1936.000000   \n",
       "mean   2013.216008     5.466705            9.368453        0.812552   \n",
       "std       4.166828     1.115711            1.154084        0.118482   \n",
       "min    2005.000000     2.375000            6.635000        0.290000   \n",
       "25%    2010.000000     4.640000            8.464000        0.749750   \n",
       "50%    2013.000000     5.386000            9.460000        0.835500   \n",
       "75%    2017.000000     6.283000           10.353000        0.905000   \n",
       "max    2020.000000     8.019000           11.648000        0.987000   \n",
       "\n",
       "       Healthy life expectancy at birth  Freedom to make life choices  \\\n",
       "count                       1894.000000                   1917.000000   \n",
       "mean                          63.359374                      0.742558   \n",
       "std                            7.510245                      0.142093   \n",
       "min                           32.300000                      0.258000   \n",
       "25%                           58.685000                      0.647000   \n",
       "50%                           65.200000                      0.763000   \n",
       "75%                           68.590000                      0.856000   \n",
       "max                           77.100000                      0.985000   \n",
       "\n",
       "        Generosity  Perceptions of corruption  Positive affect  \\\n",
       "count  1860.000000                1839.000000      1927.000000   \n",
       "mean      0.000103                   0.747125         0.710003   \n",
       "std       0.162215                   0.186789         0.107100   \n",
       "min      -0.335000                   0.035000         0.322000   \n",
       "25%      -0.113000                   0.690000         0.625500   \n",
       "50%      -0.025500                   0.802000         0.722000   \n",
       "75%       0.091000                   0.872000         0.799000   \n",
       "max       0.698000                   0.983000         0.944000   \n",
       "\n",
       "       Negative affect  \n",
       "count      1933.000000  \n",
       "mean          0.268544  \n",
       "std           0.085168  \n",
       "min           0.083000  \n",
       "25%           0.206000  \n",
       "50%           0.258000  \n",
       "75%           0.320000  \n",
       "max           0.705000  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#describe of data\n",
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "cdd70fdf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1949 entries, 0 to 1948\n",
      "Data columns (total 11 columns):\n",
      " #   Column                            Non-Null Count  Dtype  \n",
      "---  ------                            --------------  -----  \n",
      " 0   Country name                      1949 non-null   object \n",
      " 1   year                              1949 non-null   int64  \n",
      " 2   Life Ladder                       1949 non-null   float64\n",
      " 3   Log GDP per capita                1913 non-null   float64\n",
      " 4   Social support                    1936 non-null   float64\n",
      " 5   Healthy life expectancy at birth  1894 non-null   float64\n",
      " 6   Freedom to make life choices      1917 non-null   float64\n",
      " 7   Generosity                        1860 non-null   float64\n",
      " 8   Perceptions of corruption         1839 non-null   float64\n",
      " 9   Positive affect                   1927 non-null   float64\n",
      " 10  Negative affect                   1933 non-null   float64\n",
      "dtypes: float64(9), int64(1), object(1)\n",
      "memory usage: 167.6+ KB\n"
     ]
    }
   ],
   "source": [
    "# Information about the Variables\n",
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f8e4656b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>year</th>\n",
       "      <th>Life Ladder</th>\n",
       "      <th>Log GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy at birth</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "      <th>Positive affect</th>\n",
       "      <th>Negative affect</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2008</td>\n",
       "      <td>3.724</td>\n",
       "      <td>7.370</td>\n",
       "      <td>0.451</td>\n",
       "      <td>50.80</td>\n",
       "      <td>0.718</td>\n",
       "      <td>0.168</td>\n",
       "      <td>0.882</td>\n",
       "      <td>0.518</td>\n",
       "      <td>0.258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2009</td>\n",
       "      <td>4.402</td>\n",
       "      <td>7.540</td>\n",
       "      <td>0.552</td>\n",
       "      <td>51.20</td>\n",
       "      <td>0.679</td>\n",
       "      <td>0.190</td>\n",
       "      <td>0.850</td>\n",
       "      <td>0.584</td>\n",
       "      <td>0.237</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2010</td>\n",
       "      <td>4.758</td>\n",
       "      <td>7.647</td>\n",
       "      <td>0.539</td>\n",
       "      <td>51.60</td>\n",
       "      <td>0.600</td>\n",
       "      <td>0.121</td>\n",
       "      <td>0.707</td>\n",
       "      <td>0.618</td>\n",
       "      <td>0.275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2011</td>\n",
       "      <td>3.832</td>\n",
       "      <td>7.620</td>\n",
       "      <td>0.521</td>\n",
       "      <td>51.92</td>\n",
       "      <td>0.496</td>\n",
       "      <td>0.162</td>\n",
       "      <td>0.731</td>\n",
       "      <td>0.611</td>\n",
       "      <td>0.267</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2012</td>\n",
       "      <td>3.783</td>\n",
       "      <td>7.705</td>\n",
       "      <td>0.521</td>\n",
       "      <td>52.24</td>\n",
       "      <td>0.531</td>\n",
       "      <td>0.236</td>\n",
       "      <td>0.776</td>\n",
       "      <td>0.710</td>\n",
       "      <td>0.268</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Country name  year  Life Ladder  Log GDP per capita  Social support  \\\n",
       "0  Afghanistan  2008        3.724               7.370           0.451   \n",
       "1  Afghanistan  2009        4.402               7.540           0.552   \n",
       "2  Afghanistan  2010        4.758               7.647           0.539   \n",
       "3  Afghanistan  2011        3.832               7.620           0.521   \n",
       "4  Afghanistan  2012        3.783               7.705           0.521   \n",
       "\n",
       "   Healthy life expectancy at birth  Freedom to make life choices  Generosity  \\\n",
       "0                             50.80                         0.718       0.168   \n",
       "1                             51.20                         0.679       0.190   \n",
       "2                             51.60                         0.600       0.121   \n",
       "3                             51.92                         0.496       0.162   \n",
       "4                             52.24                         0.531       0.236   \n",
       "\n",
       "   Perceptions of corruption  Positive affect  Negative affect  \n",
       "0                      0.882            0.518            0.258  \n",
       "1                      0.850            0.584            0.237  \n",
       "2                      0.707            0.618            0.275  \n",
       "3                      0.731            0.611            0.267  \n",
       "4                      0.776            0.710            0.268  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the overview of the data\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0eeb776a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Country name                          0\n",
       "year                                  0\n",
       "Life Ladder                           0\n",
       "Log GDP per capita                   36\n",
       "Social support                       13\n",
       "Healthy life expectancy at birth     55\n",
       "Freedom to make life choices         32\n",
       "Generosity                           89\n",
       "Perceptions of corruption           110\n",
       "Positive affect                      22\n",
       "Negative affect                      16\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#missing data\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "42fbab18",
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop missing\n",
    "data=data.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "bbac3744",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Country name                        0\n",
       "year                                0\n",
       "Life Ladder                         0\n",
       "Log GDP per capita                  0\n",
       "Social support                      0\n",
       "Healthy life expectancy at birth    0\n",
       "Freedom to make life choices        0\n",
       "Generosity                          0\n",
       "Perceptions of corruption           0\n",
       "Positive affect                     0\n",
       "Negative affect                     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "333f3572",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Country name', 'year', 'Life Ladder', 'Log GDP per capita',\n",
       "       'Social support', 'Healthy life expectancy at birth',\n",
       "       'Freedom to make life choices', 'Generosity',\n",
       "       'Perceptions of corruption', 'Positive affect', 'Negative affect'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#columns in data\n",
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "71c940cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1708, 11)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#size of data\n",
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "35aa2c4c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop unnecessary columns\n",
    "data= data.drop(['Positive affect', 'Negative affect'], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "dc78cfbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "#rename some columns\n",
    "data.rename(columns = {'Life Ladder': 'Ladder score',\n",
    "                       'Log GDP per capita':'Logged GDP per capita'\n",
    "                       ,'Healthy life expectancy at birth':'Healthy life expectancy'},inplace='True')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "a2e4d98b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>year</th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2008</td>\n",
       "      <td>3.724</td>\n",
       "      <td>7.370</td>\n",
       "      <td>0.451</td>\n",
       "      <td>50.80</td>\n",
       "      <td>0.718</td>\n",
       "      <td>0.168</td>\n",
       "      <td>0.882</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2009</td>\n",
       "      <td>4.402</td>\n",
       "      <td>7.540</td>\n",
       "      <td>0.552</td>\n",
       "      <td>51.20</td>\n",
       "      <td>0.679</td>\n",
       "      <td>0.190</td>\n",
       "      <td>0.850</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2010</td>\n",
       "      <td>4.758</td>\n",
       "      <td>7.647</td>\n",
       "      <td>0.539</td>\n",
       "      <td>51.60</td>\n",
       "      <td>0.600</td>\n",
       "      <td>0.121</td>\n",
       "      <td>0.707</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2011</td>\n",
       "      <td>3.832</td>\n",
       "      <td>7.620</td>\n",
       "      <td>0.521</td>\n",
       "      <td>51.92</td>\n",
       "      <td>0.496</td>\n",
       "      <td>0.162</td>\n",
       "      <td>0.731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2012</td>\n",
       "      <td>3.783</td>\n",
       "      <td>7.705</td>\n",
       "      <td>0.521</td>\n",
       "      <td>52.24</td>\n",
       "      <td>0.531</td>\n",
       "      <td>0.236</td>\n",
       "      <td>0.776</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Country name  year  Ladder score  Logged GDP per capita  Social support  \\\n",
       "0  Afghanistan  2008         3.724                  7.370           0.451   \n",
       "1  Afghanistan  2009         4.402                  7.540           0.552   \n",
       "2  Afghanistan  2010         4.758                  7.647           0.539   \n",
       "3  Afghanistan  2011         3.832                  7.620           0.521   \n",
       "4  Afghanistan  2012         3.783                  7.705           0.521   \n",
       "\n",
       "   Healthy life expectancy  Freedom to make life choices  Generosity  \\\n",
       "0                    50.80                         0.718       0.168   \n",
       "1                    51.20                         0.679       0.190   \n",
       "2                    51.60                         0.600       0.121   \n",
       "3                    51.92                         0.496       0.162   \n",
       "4                    52.24                         0.531       0.236   \n",
       "\n",
       "   Perceptions of corruption  \n",
       "0                      0.882  \n",
       "1                      0.850  \n",
       "2                      0.707  \n",
       "3                      0.731  \n",
       "4                      0.776  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34a799de",
   "metadata": {},
   "source": [
    "## Description of  columns in dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a0b3e634",
   "metadata": {},
   "source": [
    "#### [Country name]:\n",
    "Name of each Country"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eaed3bed",
   "metadata": {},
   "source": [
    "#### [Life Ladder] :\n",
    "A metric measured by asking the sampled people the question: \"How would you rate your happiness on a scale of 0 to 10 where 10 is the happiest.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3faeedf8",
   "metadata": {},
   "source": [
    "#### [Logged GDP per capita ]:\n",
    "GDP per Capita of each Country in terms of Puchasing Power Parity (PPP) (in USD)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efc84bd8",
   "metadata": {},
   "source": [
    "#### [Healthy life expectancy] : \n",
    "Healthy Life Expectancy at birth are constructed based on data from the World Health Organization (WHO) and WDI."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "212984d0",
   "metadata": {},
   "source": [
    "#### [Social support ]:\n",
    "National average of the binary responses (either 0 or 1) to the question “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?”"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82245270",
   "metadata": {},
   "source": [
    "#### [Freedom to make life choices] :\n",
    "National average of binary responses to the question “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5566be3b",
   "metadata": {},
   "source": [
    "#### [Generosity ]: \n",
    "Generosity is the residual of regressing the national average of responses to the question “Have you donated money to a charity in the past month?” on GDP per capita."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53171579",
   "metadata": {},
   "source": [
    "#### [Perceptions of corruption ]:\n",
    "Perceptions of corruption are the average of binary answers to two GWP questions: “Is corruption widespread throughout the government or not?” and “Is corruption widespread within businesses or not?”"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e4c17f4",
   "metadata": {},
   "source": [
    "# General Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6652808b",
   "metadata": {},
   "source": [
    "### Take a closer look - how each factor affects overall happiness or one another\n",
    "\n",
    "Logged GDP per capita, Social Support, and Healthy Life Expectancy have a strong degree of relationship with the Ladder Score. As each of these factors increases, the overall Ladder Score increases as well.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70674a7a",
   "metadata": {},
   "source": [
    "### Factor to factor analysis shows strong degree of relationship between the following:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ebc39f9",
   "metadata": {},
   "source": [
    "##  Logged GDP per capita and Ladder Score relationship:  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c5a8d111",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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73+v1DnVrVMikUgkEolE0rBpFBHi246mI6g4VkujKAhg65H0KtVz/PhxVq1axZIlS2jWrJlapkbDww8/TFBQEACff/45ffr0ISoqigEDBnDw4EFAtfQMHz6cRx99lLCwMKKjozl58iQAkydPJjk5mcjISEaOHAlAYGAgc+bMYdCgQTzxxBNcvHiRQYMG0atXL3r06MHUqVNdilF5/vnnuf3224mMjGTAgAGkpKQAqlLXokULXnvtNXr16kXnzp3ZuHGj9by1a9cSEhJC3759mT17dpXaa/PmzfTt25eePXsSGhrKkiVLrPtef/11unXrZrUenT59msmTJwMQExNDZGQk6enp5OTk8NRTT9G7d2/Cw8OZPHmyNWX/wIEDefnllxkyZAjDhw8vt40rw8CBA/n222+tv8eMGcPSpUsB1fI1depU7r77biIiIgB488036dGjB2FhYcTGxnLt2jVAtcY98sgj3HPPPYSGhjJy5EgyMzMBKCkp4YUXXqB3795ERkYyduzYallBJRKJRCJpaAghOJCTz3upF3kt5RzvpV7kQE5+XYslkVRIo1CwMvNLMJlcC4g3mQRZ+VVbU+v3338nODiYFi1aON3/888/8+WXX5KYmMjvv//O66+/TmxsrHX/r7/+yty5czlw4ABDhw7l3//+NwDz58+ne/fuJCUl8c0331iPP3PmDFu2bGHFihU0bdqUDRs28Ntvv7F//35OnjzJmjVrKpR5xowZ7Nmzh6SkJP7617/y7LPPWvdlZGTQq1cvfvvtNz766CPrvvT0dJ566inWr1/PL7/8goeHR7l1bNq0yc7V7oMPPgAgKiqKHTt2sG/fPhITE9Hr9Vy4cIHMzEzefvttfv/9d5KSkti5cye33HIL8+fPB2Dnzp0kJSXRqlUrpk2bRv/+/dm9ezd//PEHBoOBjz76yFp3UlIS33//PZs3by63jZ0xd+5cO7l37txZYXsC7Nixg9WrV3Po0CH++9//smTJEn7++WcOHDiAj48PL730kvXYn376iSVLlnDw4EHat2/Pyy+/DMBbb72Fr68vu3fvJikpiR49ehAfH+9S/RKJRCKRNHRS8gp5cN9xxiSd4OMz6ay8kMHHZ9IZk3SCB35P4Xh+YV2LKJGUSaPwX2rm7YZG41qmQY1Goam32w2RY/369fzxxx/06dPHuu3y5cvWmKI777yT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Onbv3l0tWSUSiUTSOAn18+aZwNbog9vxTGBrqVxJGgSNw4IlhGq5Ks4HN6/S+xWNur04H775B0z8vspVTZo0iatXr/LLL7/QrFkzTCYTa9as4erVqzRt2rTq13ATYlGwnn766Rotd9++fZw6dYpDhw7VaLk3G0uXLqVFixZ06dKl1D6DwcBDDz3Epk2b6NmzJwBHjx7Fx8enwnMdWbduHdHR0fTu3btmL0AikUgkEomkHtI4LFgX96tugTqP8o/TeajHXdhfpWqOHz/OqlWrWLJkCc2aNQNAo9Hw8MMPExQUBMCbb75Jjx49CAsLIzY2lmvXrgFqyu5HH32U++67j86dO/PII4+wb98+Bg8eTFBQEM8995y1noEDB/L888/Tv39/br31Vt566y2+/PJLYmJi6NixI19++aX12O+//56oqCjCw8MZMGAAycnJAGzbto3IyEimTJlCREQEPXr0YO/evQC8+OKLvPHGGwB88803KIpCSopq2Rs3bhyff/45AHv27GHw4MFER0cTFRXFmjVrAHVwPnz4cKKjo+nRowexsbHk5+eXaq/JkyeTnJxMZGQkI0eOtG5fs2YNMTEx3Hbbbbz++utltreztkxOTiY2NpZTp04RGRnJsmXLyr1nAwcOZMaMGfTr149OnToxefJk675z584xZMgQwsPDeeCBB7hy5Yp1X05ODk899RS9e/cmPDycyZMnU1JSYi3zmWeeYeDAgQQHB/P8889jWdD74sWLPPLII9bzXnvtNWuZgYGB6PV6p9eenJxMnz59iIqKIjY2lsLCQuu+qpS5cOFC9u7dy9SpU4mMjGTjxo127ZKTk0NOTg5t2rSxbuvatSu+vr5Ozz1w4AD9+vUjKiqK7t27W5+fjRs38s033zB37lwiIyNZuHAhAJ9//rn1egYMGGC1KO7atYtevXoRGRlJaGgon3zySbn3TyKRSCQSiaTeIYRoEH/t2rUTjhgMBpGcnCwMBkOpfXZsf1OIf7UX4q0uFf/9q716fBX46quvRHh4eJn7N27cKEJCQkRmZqYQQoinnnpKTJkyRQghRHx8vOjcubPIysoSBoNBhIeHi2HDhonCwkKRm5srWrZsKY4ePSqEEGLAgAHikUceEUajUZw7d054enqKl19+WQghxK+//iratGkjhBDi0qVLIiAgQOzfv18IIcTy5ctFjx49hBBC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Gzdu5PDhw3zxxRd8/vnnbNu2jZ9//pnXXnuN3NxcjEYjjz32GPPmzWPPnj388ssvzJ8/n99//91aTlFREdu2bStludq2bRtz5szhv//9L3/88QeJiYm0atUKrVbLypUr2bt3LwcPHsTf399OabNlxowZ7Nmzh6SkJP7617/y7LPPWvdlZGTQq1cvfvvtNz766CPrvvT0dJ566inWr1/PL7/8gkc5PtOgWvkmTpxIVFQUzZo1Y/PmzXb727Zty9NPP018fHyZZZw/f55HHnmERYsW0bVr13Lrk0gkEolEUnk2Z2RXmGwCVEsWQrApI7vGZWgVGMQdo/7MwMef5I5Rf5YxVxKnNAoFqy7WMHj99dd57733yMjIsNu+adMmq/tYq1atGDVqlN2AfsKECRV2HLGxsQC0bNmSoKAgTp065fQ4i+IAMGnSJJYuXYrRaLTuf/DBB/Hw8MDb25uuXbtyzz33oNFoaNeuHc2aNePs2bMcPXqUQ4cOMXbsWCIjI4mJiSEnJ4fk5GRrOZY6HPnuu+94/PHHadOmDQDe3t54e3sjhODdd9+lZ8+ehIeH891335GUlOS0jB9++IG+ffsSGhrKrFmz7I7z8fHhgQceAKBv375WC92uXbuIioqyKjpPP/10mW155coVfvzxRx599FFrOzlzp5wxYwYbNmzgyJHS62IXFxczevRonnzySas8EolEIpFIaparJUZcG82BCcgsMVZ4nERyI2gUWQTrYg2DoKAgHn30UacufM5iwCz4+lbs0+zpeT0zjlartUuoYSEpKYkDBw7w9NNPW8u/cuUK33//Pffee6/TcpyVqygKLVq0KFMBclVmW1auXMn27dtJTEzEz8+PDz74gMTExFLHnTlzhqlTp7J7926CgoLYv38/gwcPtu53lNeiPFa0doctn3/+OQaDgcjISACMRiMZGRlkZGQQEBBgPa5Jkyb83//9Hy+++CJarX3a2n/84x80b96cmTNnulyvRCKRSCSSytHcTeuyZUADNCsnzbxEciNpFBas23pGg1LxwLum1zB49dVXWb58uV3sztChQ61xV5cvX+Y///mPndJgi7+/v132usqwcOFCpk2bxunTp0lNTSU1NZV58+ZVOtlF165d8fb2ZtmyZdZtx48f5+rVqxWee//997Ns2TIuXrwIQH5+Pvn5+WRmZhIQEICfnx85OTlO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a3V3jaBYBF2/Ciq4K8uirVUreq22FmjTBgotPSZNkzaZSeb0ef/++GTSnDOTTjKHPO/ceksz88nMe9JJ+Lw+r9f79UqwevVqvve97/He97532HGf/exnWb16Na973eu46qqrePe7382ll14KwI033sj73/9+jDG8+uqrXH311ezfv79/2nZnZyfbtm2jra2NCy+8kNtvv52zzjqL2267jU984hMAPPfcc3zkIx8ZFAzGYrH+YLCrq4tnn32WYDBIT08PV199NW95y1u44oorANixYwc//elPCQQCrF69mscff5z3v//9w17Hiy++yGuvvcZNN91EMpnkH/7hH7jjjjv40Ic+xL59++ju7uYLX/hC//E7duzg6aefxu/38+qrr/bffuLECf74j/+YJ554gquvvhrHcejs7ATgr/7qr/j85z8PuNmyv/zLv+THP/5xRv8eIjJAyzPu5W0zznU543GPa/kvWDB1g3VFshHbvXvE4GUoEwwS27uX3t27Kaury1kWN5M5VN/d8gNuTTn4BtyXcCzJlGVQPGgMWMslh17kwGz34mu6A+CLh0+yYuFMtuxpxUJ/IDmjzH86C2YGP5RjIRpPURUK0HDVBXx8/+GM5m9hDNdXzxjz+ykyXUyLDFYi3oGbvM6EQyIxsWnxHo+Hxx9/nG3btnHTTTfxq1/9ihUrVvDSSy9l9PXr16/H7/cTCoVYt24dzz777IjHffKTn+Tll1/mzjvv5MSJE1xzzTV897tuWeMrr7zCW9/6VlauXMm73vUujh8/zv79+/u/9oMf/CAAc+bM4fzzz+eVV14BYPv27Vx77bWsXLmSu+66i127dg3KPN1xxx39f+/p6eHOO+/koosu4qqrrmL//v3s2LGj//5bbrmF8vJyvF4vV155JS+//PKIr+PrX/86H/rQh/B6vbz97W9n37597N69e9Tvz2233dYfKA7061//muXLl3P11VcD7r/D7NmzAXjmmWd4/etfz8qVK/nc5z43aJ0ikoXoCbcUKhPWgZ7OSV2OyJno3roVxgka4HRA0v3zn+csizt0DpVnyBrSWahgtIveeHLQffHkyD+DxlrCscigx7DA5uZWADqiCRzndMrK6zFUhQL9mayBfwBmlft5bMPruOns2dSFy+h1xt6vH3Msy8JlU9bgwlpL24Eu/uc/X+WX39vL//znq7Qd6JqS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IiIiIiI5ogBLREREREQkRxRgiYiIiIiI5IgCLBERERERkRxRgCUiIiIiIpIjatMuIhmx1tLdvYvj7VtIxDvwB6qoqb6OysoV+V6aiIiISMFQgCUi44pEXmZ38z1EIntxh5M7gIcDBx4iHF7Csrr7CYfPz/cyRURERPJOJYIiMqZI5GW277iNSPcejAng9Zbj9YbxessxJkCkew/bd6wjEtmX76WKiIiI5J0CLBEZlbWW3c33kEpG8XjLMWbwrwxjPHi85aSSUXY3352nVYqIiIgUDgVYIjKq7u5dRCJ7MZ6yMY8zniCRyF66unZN0cpERERECpMCLBEZ1fH2LVhrMcaMeZwxHqy1HG/fPEUrExERESlMCrBEZFSJeAduQ4tMOCQSJydzOSIiIiIFTwGWiIzKH6gi818THvz+mZO5HBEREZGCpwBLREZVU30dxpi+1uyjs9bBGENN9ZopWpmIiIhIYVKAJSKjqqhYTji8BOv0jnmcdWKEw0uorFw+RSsTERERKUwKsERkVMYYltXdj9cXwkn1YO3g/VjWOjipHry+MMvq7s/TKkVEREQKhwIsERlTOHw+l656lHDFUqyNk0r1kEpFSKV6sDZOuGIpl676FuHw+fleqoiIiEjemfH2VhSKRYsW2YMHD+Z7GSLTWlfXLo63byaROInfP5Oa6jUqCxQREZFpxRhzyFq7aLT7fVO5mDRjzDPAfNz+z13AX1hrd+RjLSKSucrK5QqoRERERMaQlwALeK+1thPAGPMu4BvAZXlai4iIiIiISE7kZQ9WOrjqM5PMJ5mKiIiIiIgUrHxlsDDGPAKkh+bcNML9nwA+kf585kwNMBURERERkcKW9yYXxpj1wJ9Ya9821nFqciEiIiIiIvk2XpOLvLdpt9ZuAtYYY6rzvRYREREREZEzMeUBljFmhjFm4YDP/xhoB05M9VpERERERERyKR97sGYCjxtjynGbW7QB77D5rlUUERERERE5Q1MeYFlrXwOunOrnFRERERERmWx534MlIiIiIiJSKhRgiYiIiIiI5IgCLBERERERkRxRgCUiIiIiIpIjCrBERERERERyRAGWiIiIiIhIjijAEhERERERyREFWCIiIiIiIjmiAEtERERERCRHFGCJiIiIiIjkiAIsERER8XOofwAADaNJREFUERGRHFGAJSIiIiIikiMKsERERERE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\n",
      "text/plain": [
       "<Figure size 1040x560 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig_dims = (13, 7)\n",
    "fig, ax = plt.subplots(figsize=fig_dims,  dpi=80)\n",
    "\n",
    "plt.style.use('tableau-colorblind10') #style.available\n",
    "groups = data2021.groupby(\"Regional indicator\")\n",
    "for name, group in groups:\n",
    "    plt.plot(group[\"Logged GDP per capita\"], group[\"Ladder score\"], marker=\"o\", linestyle='', label=name, ms=10, alpha=0.9)\n",
    "    \n",
    "ax.set_xlabel('GDP per capita', color='#006680', fontsize=15, fontweight='bold')\n",
    "ax.set_ylabel('Ladder Score', color='#006680', fontsize=15, fontweight='bold')\n",
    "ax.set_title('GDP per capita against Happiness Score per Continent', color='#006680', fontweight='bold', fontsize=20);\n",
    "\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "raw",
   "id": "5d96e55c",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "f4f4cbf6",
   "metadata": {},
   "source": [
    "## Social_Support and Ladder Score relationship:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b636787f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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qoqSkhIWL15cqzwsFkud3M/ixYsZNmxYufq89dZbzJ07l5SUFGJiYsqV3dB9QiKRSCSShiQlJQXAJyHBNb0njn2lgoJU33qbzeb8A1XLVV9+SH4hYVVqrxwoigZrm+6VCkYaBZoH+qHVKCCg2RVd+K3bdH7p8ygpPe+nWZSef42JItjPt0iDDuojuIfEnSYhYO3OLLD7W/ki2avpa8N9993Hu+++C8C5c+fYtm0bY8aMcZ5funQpt912m/P3U089Rbdu3Rg6dChfffWVW16VnXPlm2++4brrrqNfv34MGDCALVu2VLveL7/8MldffTV9+/alf//+/PLLL4C6r8Sf//xnevfuTVxcHKNGjQJgxowZpKamotfrGT9+fLn89uzZw+DBg4mPj6d3794899xzznP33HMPDz74IImJifTo0YMJEyZQWlrqbLPbbruN3r17M3r0aA4fPlxpvY1GI88884xXlffp06e55ZZbiImJITo62k2AioyMZP78+QwfPpy77767wvtZuXIlgwYN4sorr2TevHkV1iM/P5+1a9fywQcfsG/fPtLS0gCYOXMmP/zwA7NmzWLQoEGA2tdeeuklhg0bxhNPPFGuTyxZsgS9Xk9cXBwJCQmkp6djsVgYPXo0CQkJ9OnTh0mTJkk1v0QikUguCwoLC9022a0Mm81GUVFRhecd+0p17dqVdu3aERYWRrt27ejatSvdu3evtyAPLePHgqJUqcUSwgaKQkCH8qHaPVGFLC2B/gq9rghiaM/W3BDTllljuvLk2G50aKFr8OAekvI0iSAXRWYrVZjjOhECzpfUTrIfMmQIr732GllZWaxevZrbb7+9ws68evVqvvzyS1JSUggKCuKWW27x6ZwrR44cwWg0sm7dOkJDQzl8+DBDhw4lPT3dq6p9wYIFLFq0yPn7jTfeYNCgQUyePJmHH34YgG3btjF16lT27t3LunXryMvLIzU1FYDcXHUju7feeotHH32UnTt3eq1XZGQk69evJzAwkOLiYgYNGsT111/v1OalpKSwYcMGAgICGDJkCCtXruTOO+9k7ty5hIaGkpqaytmzZ4mPj+eOO+6osL3j4+MZMmQIr7zyCrNnz3Y7N3PmTHr16sXnn39OdnY2/fr1Q6/X079/fwCOHz/uNOvbvHmz1/vJz89n69atnDlzhm7dunHvvffSqVOncvVYvnw5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Pfccl112GV/4whdyfi+f//znJ5QIPv7440X9DH71q1/xgx/8gOeff55f/OIX3HnnnTz22GM899xzNDY2ctNNN40d++ijj3LnnXeybds2WltbufnmmwH40pe+xLx583jqqafYsmULZ511lgZliohkWLFiBR0dHdx8881cc801XHrppVxzzTV84hOfoKOjo2YDiOlW6v6iWjPdJaDl6EK5trWZm688m3+87tXcfOXZnHfiorxbMWKx2FjA2Nzsgglj00v+DGCJRLJ/z8ZAXbLTYikljl+/6wfJIcJLXTAVj0B81F2mMlelDBkuUikllbffcGHR9zud3RylvErJYH0SWJ7l+ruA9wGXAwXfBVtrDxljHsR1EPy5MWYlcArwYglrKUmpw0iPZ3jpJZdcwsMPP0xbWxuvfe1rAfi93/s9nnzySR5++GEuueQSAH784x/zxBNP8I//+I+AK2MMBAI0NTWxevVq/vRP/5QrrriCq6++mtbW7HPMct1Hyrvf/e4JJQBveMMbWLlyJQCve93ruPXWW3N+H1MtEXzHO94xVq7wwAMP8Cd/8idjmaf//b//N+985zvHjr3mmmtYunQpAH/2Z3/GO97xjrHva3BwkB/84AcARCIRTj311JLXIiIyFxSaMTSXZL75zlTOFuepx5vq/KZcprsEdDq6UObbipHKrqZu8/v9+P1+YpEjjNavGJv0lcpkJRIWj2fi92twP1eYQolj3zAnr2t3c64Ovei6Bfob3J6rMpYFpq83NZ/qslct5+GXDnB4OILHmKLmo+UzXaWHUn6lvAJcjet3+SfA99Ou/23ycmUJ9/XnwDeNMV8A4sCfTWeji6amJvr6sg/zS5e6valp6m0tL7nkEr75zW/S1tbGm970JgDe9KY38dBDD/HQQw+NvRhaa/nxj3+ctUTviSee4PHHH+fhhx/mwgsv5O677+aNb3xj1vXmug9g0otiegt6r9dLLDZ5w2ghPp9vQoYvs+ti+mNm+3nn+/mnbrPW8rWvfY1LL7205PWJiMjcNdX9RVMxbfObmN4S0OkoQcy3FSNbVVAwGCRy7BieWIiErwHseEYtkYjj8Yy/PY1bw6IG71jQOuUSx/krpiWgSpdrPhXAwvoAl5x+Aqctaco7Hy2f6So9lPIrpYtg6ln5w4zrUwWei4u9I2vtTmvtm62151hr11prf1TCOkpW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KnN1/vvdcSIiMuMUGMQhIiIiZde4BNa1uzlXh1503QL9DW7PlcoCRURmtKoEWMaYXwLLgARwDPhra+2WaqxFRESkauavUEAlIjLLVCuD9Q5r7QCAMeZtwDeBC6q0FhERERERkbKoyh6sVHCVtACXyRIREREREZnRqrYHyxhzF3BJ8tMrs9z+EeAjqc8XLFhQoZWJiIiIiIhMjbGFhh1O9wKMeS9wvbX2qnzHtba22u7u7gqtSkREREREZDJjzD5rbWuu26vept1a+23gEmNMS7XXIiIiIiIicjwqHmAZY5qMMSvSPv9DoB84XOm1iIiIiIiIlFM19mAtAP7TGFOPa27RB1xjq12rKCIiIiIicpwqHmBZa/cCr6n044qIiIiIiEy3qu/BEhERERERmS0UYImIiIiIiJSJAiwREREREZEyUYAlIiIiIiJSJgqwREREREREykQBloiIiIiISJkowBIRERERESkTBVgiIiIiIiJlogBLRERERESkTBRgiYiIiIiIlIkCLBERERERkTJRgCUiIiIiIlImCrBERERERETKRAGWiIiIiIhImSjAEhERERERKRMFWCIiIiIiImWiAEtERERERKRMFGCJiIiIiIiUiQIsERERERGRMlGAJSIiIiIiUiYKsERERERERMpEAZaIiIiIiEiZKMASEREREREpE1+1FzDTWGs5ePAgO3fuJBQK0dDQwKpVq1i6dGm1lyYiIiIiIlWmAKsE/f39bNq0ib6+PgASiQQej4ennnqKJUuWsH79elpaWqq8ShERERERqRaVCBapv7+fjRs30tfXh9frxefzEQgE8Pl8eL1e+vr62LhxI/39/dVeqoiIiIiIVIkCrCJYa9m0aRPRaBSfz4cxZsLtxhh8Ph/RaJRNmzZVaZUiIiIiIlJtCrCKcPDgwbHMVT6pTFZvb2+FViYiIiIiIrVEAVYRdu7cCTApc5UpdXvqeBERERERmVsUYBUhFAqRSCSKOjaRSDAyMjLNKxIRERERkVqkAKsIDQ0NeDzF/ag8Hg/19fXTvCIREREREalFCrCKsGrVKsA1u8gndXvqeBERERERmVsUYBXhhBNOYMmSJcTj8bzHxeNxlixZoqHDIiIiIiJzlAKsIhhjWL9+PX6/n1gsNimTZa0lFovh9/tZv359lVYpIiIiIiLVpgCrSC0tLVx//fVjmaxYLEYk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\n",
      "text/plain": [
       "<Figure size 1040x560 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig_dims = (13, 7)\n",
    "fig, ax = plt.subplots(figsize=fig_dims,  dpi=80)\n",
    "\n",
    "plt.style.use('tableau-colorblind10') #style.available\n",
    "groups = data2021.groupby(\"Regional indicator\")\n",
    "for name, group in groups:\n",
    "    plt.plot(group[\"Social support\"], group[\"Ladder score\"], marker=\"o\", linestyle='', label=name, ms=10, alpha=0.9)\n",
    "    \n",
    "ax.set_xlabel('Social Support', color='#006680', fontsize=15, fontweight='bold')\n",
    "ax.set_ylabel('Ladder score', color='#006680', fontsize=15, fontweight='bold')\n",
    "ax.set_title('Social Support Status against Happiness Score per Continent', color='#006680', fontweight='bold', fontsize=20);\n",
    "\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86097767",
   "metadata": {},
   "source": [
    "## Healthy Life Expectancy and Ladder Score relationship:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "38366b42",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Grp8ww030NfXN/c3VnKy1LeFDA0N8cADDzA0NITP5yMQCEy++Xw+hoaG2LlzJ8PDw+VeatXKOcAynd2twC+BdwBNTO0ErQPeD1yZ403tw+2/+i6AtfZJ4AXgrFzXkq94fPYa6YUen1RTU8OrX/1q7rvvPiYmJujt7eXyyy9n3759RCKRyUxWKBTiE5/4BBdddBGPPvooTz75JLFYjG9/+9scP36cr33ta/zqV79i165dPPTQQ7S2tvJP//RPADz00EPs2rWLk046KettJO3atYu7776be+91ycFHHnmEL33pSzz11FNccsklfPnLX876tXzpS1+aViL40EMP5fQ9+MUvfsG///u/85vf/Iaf/OQnfOc73+HBBx/kqaeeoqGhgc985jOTxz7wwAN85zvf4emnn6a9vZ3rrrsOgK9+9as0Njby6KOPsmvXLs4666wZHZxERKSytbW10dXVxXXXXcdb3vIW3vCGN/CWt7yFz372s3z4k934t3wQGlrBxiAegfiEe29j0NCKedk1dP3RG2aU4dVHhji3/wHs6KArATRgfC5rZIz7Gx6ZmGBqrhZgLbHI+OSHfr8fm9rLMMt+MZ/1GAk009rWMevXurl9BddddjZff/vLue6ysyf3XKUPXc60RSEUCjExMcGNN96Yx3dXsinVtpBKZa3l8ccfJxaL4ff7M379fr+fWCzGY489VqZVVr98arg+B6zNcPmtwAeAS4E5z4KttUeMMffiOgj+2BizDjgVeC6PteTF7585ab2Qx6e6+OKLuf/+++no6OBVr3oVAK94xSt45JFHuP/++7n44osB+NGPfsTDDz/M17/+dcC9mhIKhWhubua0007jPe95D29605t485vfTHt7e8b7ynYbSe9973unDXp8zWtew7p16wB49atfzbe+9a2sX8d8SwTf+c53ksw2/uxnP+Pd7343LS0tAPzZn/0Z73rXuyaPfctb3kJraysAH/zgB3nnO985+XUNDQ3x7//+7wBEIhE2btyY91pEZHGz1nJ8LM6BwQjhmKU2YDi5JcSKBpUfLyZZZzI1rIYtnTDcB0eeg+gYBOth1SZocuVymxqYXobneZxz5DH8XpSD+55j4/mvdyV5KTdrTKJBhmVaieDx3qnTjJqAj1FfkJCXyHSlF7hYi896xH0Bnl6+mS/NswtgLkOXgclMVnq3wkpgrWVX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qSLag1UI5e5GJiKLT+rAY5Ohi2DSYisxrva9ZSIiInNQgFVI5epGJiKLT7LEOHXgcbY5WIupxLia95aJiIjkQCWCIiLlUo0lxtW6t0xERCRHymCJiJRTtZUYa3yFiIgscQqwREQqQbWUGFfr3jIREZEcqURQREQKR+MrRERkiVOAJSIihVWNe8tERERypBJBEREpvGrbWyYiIpIjBVgiIlI81bK3TEREJEcqERQRERERESkQBVgiIiIiIiIFogBLRERERESkQBRgiYiIiIiIFIgCLBERERERkQJRgCUiIiIiIlIgatMuIoVhLYwc1MwjERERWdIUYInIwo0ehmd/CKP9LtDCAgZefAAaWuGMt7vBsyIiIiJVTiWCIrIwo4fhiZtdcGUC4A+Bv8a9NwF3+RM3wehAuVcqIiIiUnQKsERk/qx1mat4BHxBMGb69ca4y+MRePb28qxRREREpIQUYInI/I0cnMpczSaZyRruK826RERERMpEAZaIzN+R51wWKz1zlc4Yd9yR50qzLhEREZEyUYAlIvMXHcM1tMiFTRwvIiIiUr0UYInI/AXrgTmyV5NM4ngRERGR6qUAS0Tmb9WmqfK/2STLCFdtKs26RERERMpEAZaIzF/jWjfnysZmP87G3HEaOiwiIiJVTgGWiMyfMW6IsD8EXnRmJstad7k/5I4TERERqXIKsERkYRpWw8uumcpkxSMQn3Dvk5mrl13jjhMRERGpcnMMrxERyUHDatjS6eZcHXnOdQsM1rs9VyoLFBERkSWkLAGWMeanwBrAA4aBj1hrd5VjLSJSQE1tCqhERERkSStXBuud1tpBAGPM24CbgfPLtBYREREREZGCKMserGRwlbAMl8kSERERERFZ1Mq2B8sYcytwceLDyzJc/3Hg48mPly1bVqKViYiIiIiIzI+xcw0ILfYCjHk/cKW19vLZjmtvb7e9vb0lWpWIiIiIiMhMxpgD1tr2bNeXvU27tfZfgIuNMSvLvRYREREREZGFKHmAZYxpNsa0pXz8h8BR4Fip1yIiIiIiIlJI5diDtQz4gTGmDtfcYgB4iy13raKIiIiIiMgClTzAstbuB15Z6vsVEREREREptrLvwRIREREREakWCrBEREREREQKRAGWiIiIiIhIgSjAEhERERERKRAFWCIiIiIiIgWiAEtERERERKRAFGCJiIiIiIgUiAIsERERERGRAlGAJSIiIiIiUiAKsERERERERApEAZaIiIiIiEiBKMASEREREREpEAVYIiIiIiIiBaIAS0REREREpEAUYImIiIiIiBSIAiwREREREZECUYAlIiIiIiJSIAqwRERERERECkQBloiIiIiISIEowBIRERERESkQBVgiIiIiIiIFogBLRERERESkQBRgiYiIiIiIFEig3AsQKRRrLYcPH2bPnj2MjY1RX1/Phg0baG1tLffSRERERGSJUIAlVeHo0aPs2LGDgYEBADzPw+fz8eijj7J69Wq2bt3KypUry7xKEREREal2KhGURe/o0aPcdtttDAwM4Pf7CQQChEIhAoEAfr+fgYEBbrvtNo4ePVrupYqIiIhIlVOAJYuatZYdO3YQjUYJBAIYY6Zdb4whEAgQjUbZsWNHmVYpIiIiIkuFAixZ1A4fPjyZuZpNMpPV399fopWJiIiIyFKkAEsWtT179gDMyFylS16fPF5EREREpBgUYMmiNjY2hud5OR3reR7j4+NFXpGIiIiILGUKsGRRq6+vx+fL7WHs8/moq6sr8opEREREZClTgCWL2oYNGwDX7GI2yeuTx4uIiIiIFIMCLFnUTjrpJFavXk08Hp/1uHg8zurVqzV0WERERESKSgGWLGrGGLZu3UowGCQWi83IZFlricViBINBtm7dWqZVioiIiMhSoQBLFr2VK1dy5ZVXTmayYrEYkUiEWCw2mbm68sorWblyZbmXKiIiIiJVzsy1d6VStLe3297e3nIvQypcf38/e/bsYXx8nLq6OjZs2KCyQBEREREpGGPMAWtte7brA6VcjEixtba2KqASERERkbJRiaCIiIiIiEiBKMASEREREREpEAVYIiIiIiIiBaIAS0REREREpEAUYImIiIiIiBSIAiwREREREZECUYAlIiIiIiJSIAqwRERERERECkQBloiIiIiISIEowBIRERERESkQBVgiIiIiIiIFYqy15V5DTowxE8BAudexiDQCI+VehJSVHgNLm37+oseA6DGwtOnnXzyrrbU12a5cNAGW5McY02utbS/3OqR89BhY2vTzFz0GRI+BpU0///JRiaCIiIiIiEiBKMASEREREREpEAVY1etvy70AKTs9BpY2/fxFjwHRY2Bp08+/TLQHS0REREREpECUwRIRERERESkQBVgiIiIiIiIFogCr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\n",
      "text/plain": [
       "<Figure size 1040x560 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig_dims = (13, 7)\n",
    "fig, ax = plt.subplots(figsize=fig_dims,  dpi=80)\n",
    "\n",
    "plt.style.use('tableau-colorblind10') #style.available\n",
    "groups = data2021.groupby(\"Regional indicator\")\n",
    "for name, group in groups:\n",
    "    plt.plot(group[\"Healthy life expectancy\"], group[\"Ladder score\"], marker=\"o\", linestyle='', label=name, ms=10, alpha=0.9)\n",
    "    \n",
    "ax.set_xlabel('Healthy Life Expectancy', color='#006680', fontsize=15, fontweight='bold')\n",
    "ax.set_ylabel('Ladder score', color='#006680', fontsize=15, fontweight='bold')\n",
    "ax.set_title('Healthy Life Expectancy against Happiness Score per Continent', color='#006680', fontweight='bold', fontsize=20);\n",
    "\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ff2dc5d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#total of countries in regional indicator\n",
    "total_countries=data2021.groupby('Regional indicator')['Country name'].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "3995a2bd",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Regional indicator\n",
       "Central and Eastern Europe            17\n",
       "Commonwealth of Independent States    12\n",
       "East Asia                              6\n",
       "Latin America and Caribbean           20\n",
       "Middle East and North Africa          17\n",
       "North America and ANZ                  4\n",
       "South Asia                             7\n",
       "Southeast Asia                         9\n",
       "Sub-Saharan Africa                    36\n",
       "Western Europe                        21\n",
       "Name: Country name, dtype: int64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "total_countries"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99c4f0d9",
   "metadata": {},
   "source": [
    "## mean of corruption "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "ec7715a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "corruption=data2021.groupby('Regional indicator')[['Perceptions of corruption']].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "887df05c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Regional indicator</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Central and Eastern Europe</th>\n",
       "      <td>0.850529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Commonwealth of Independent States</th>\n",
       "      <td>0.725083</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>East Asia</th>\n",
       "      <td>0.683333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Latin America and Caribbean</th>\n",
       "      <td>0.792600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Middle East and North Africa</th>\n",
       "      <td>0.762235</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>North America and ANZ</th>\n",
       "      <td>0.449250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>South Asia</th>\n",
       "      <td>0.797429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Southeast Asia</th>\n",
       "      <td>0.709111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sub-Saharan Africa</th>\n",
       "      <td>0.765944</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Western Europe</th>\n",
       "      <td>0.523095</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                    Perceptions of corruption\n",
       "Regional indicator                                           \n",
       "Central and Eastern Europe                           0.850529\n",
       "Commonwealth of Independent States                   0.725083\n",
       "East Asia                                            0.683333\n",
       "Latin America and Caribbean                          0.792600\n",
       "Middle East and North Africa                         0.762235\n",
       "North America and ANZ                                0.449250\n",
       "South Asia                                           0.797429\n",
       "Southeast Asia                                       0.709111\n",
       "Sub-Saharan Africa                                   0.765944\n",
       "Western Europe                                       0.523095"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corruption"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "a372524c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.rcParams['figure.figsize']=(12,6)\n",
    "plt.title=('Perceptions of corruption in regions')\n",
    "plt.xlabel=('Regions')\n",
    "plt.ylabel=('Perceptions of corruption')\n",
    "plt.xticks(rotation=45, ha='right') \n",
    "plt.bar(corruption.index,corruption['Perceptions of corruption']);\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a738102",
   "metadata": {},
   "source": [
    "## Top 10 and Bottom 10 countries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "8a4ca470",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sort the dataset in 2021 \n",
    "top_10 = data2021.sort_values(by='Ladder score', ascending=False).head(10)\n",
    "bottom_10 = data2021.sort_values(by='Ladder score', ascending=False).tail(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "91a59aeb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>Regional indicator</th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Finland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.842</td>\n",
       "      <td>10.775</td>\n",
       "      <td>0.954</td>\n",
       "      <td>72.0</td>\n",
       "      <td>0.949</td>\n",
       "      <td>-0.098</td>\n",
       "      <td>0.186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Denmark</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.620</td>\n",
       "      <td>10.933</td>\n",
       "      <td>0.954</td>\n",
       "      <td>72.7</td>\n",
       "      <td>0.946</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.179</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Switzerland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.571</td>\n",
       "      <td>11.117</td>\n",
       "      <td>0.942</td>\n",
       "      <td>74.4</td>\n",
       "      <td>0.919</td>\n",
       "      <td>0.025</td>\n",
       "      <td>0.292</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Iceland</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.554</td>\n",
       "      <td>10.878</td>\n",
       "      <td>0.983</td>\n",
       "      <td>73.0</td>\n",
       "      <td>0.955</td>\n",
       "      <td>0.160</td>\n",
       "      <td>0.673</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Netherlands</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.464</td>\n",
       "      <td>10.932</td>\n",
       "      <td>0.942</td>\n",
       "      <td>72.4</td>\n",
       "      <td>0.913</td>\n",
       "      <td>0.175</td>\n",
       "      <td>0.338</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Norway</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.392</td>\n",
       "      <td>11.053</td>\n",
       "      <td>0.954</td>\n",
       "      <td>73.3</td>\n",
       "      <td>0.960</td>\n",
       "      <td>0.093</td>\n",
       "      <td>0.270</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Sweden</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.363</td>\n",
       "      <td>10.867</td>\n",
       "      <td>0.934</td>\n",
       "      <td>72.7</td>\n",
       "      <td>0.945</td>\n",
       "      <td>0.086</td>\n",
       "      <td>0.237</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Luxembourg</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.324</td>\n",
       "      <td>11.647</td>\n",
       "      <td>0.908</td>\n",
       "      <td>72.6</td>\n",
       "      <td>0.907</td>\n",
       "      <td>-0.034</td>\n",
       "      <td>0.386</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>New Zealand</td>\n",
       "      <td>North America and ANZ</td>\n",
       "      <td>7.277</td>\n",
       "      <td>10.643</td>\n",
       "      <td>0.948</td>\n",
       "      <td>73.4</td>\n",
       "      <td>0.929</td>\n",
       "      <td>0.134</td>\n",
       "      <td>0.242</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Austria</td>\n",
       "      <td>Western Europe</td>\n",
       "      <td>7.268</td>\n",
       "      <td>10.906</td>\n",
       "      <td>0.934</td>\n",
       "      <td>73.3</td>\n",
       "      <td>0.908</td>\n",
       "      <td>0.042</td>\n",
       "      <td>0.481</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Country name     Regional indicator  Ladder score  Logged GDP per capita  \\\n",
       "0      Finland         Western Europe         7.842                 10.775   \n",
       "1      Denmark         Western Europe         7.620                 10.933   \n",
       "2  Switzerland         Western Europe         7.571                 11.117   \n",
       "3      Iceland         Western Europe         7.554                 10.878   \n",
       "4  Netherlands         Western Europe         7.464                 10.932   \n",
       "5       Norway         Western Europe         7.392                 11.053   \n",
       "6       Sweden         Western Europe         7.363                 10.867   \n",
       "7   Luxembourg         Western Europe         7.324                 11.647   \n",
       "8  New Zealand  North America and ANZ         7.277                 10.643   \n",
       "9      Austria         Western Europe         7.268                 10.906   \n",
       "\n",
       "   Social support  Healthy life expectancy  Freedom to make life choices  \\\n",
       "0           0.954                     72.0                         0.949   \n",
       "1           0.954                     72.7                         0.946   \n",
       "2           0.942                     74.4                         0.919   \n",
       "3           0.983                     73.0                         0.955   \n",
       "4           0.942                     72.4                         0.913   \n",
       "5           0.954                     73.3                         0.960   \n",
       "6           0.934                     72.7                         0.945   \n",
       "7           0.908                     72.6                         0.907   \n",
       "8           0.948                     73.4                         0.929   \n",
       "9           0.934                     73.3                         0.908   \n",
       "\n",
       "   Generosity  Perceptions of corruption  \n",
       "0      -0.098                      0.186  \n",
       "1       0.030                      0.179  \n",
       "2       0.025                      0.292  \n",
       "3       0.160                      0.673  \n",
       "4       0.175                      0.338  \n",
       "5       0.093                      0.270  \n",
       "6       0.086                      0.237  \n",
       "7      -0.034                      0.386  \n",
       "8       0.134                      0.242  \n",
       "9       0.042                      0.481  "
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Top 10 of happiest country\n",
    "top_10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "eea6f650",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>Regional indicator</th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>139</th>\n",
       "      <td>Burundi</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.775</td>\n",
       "      <td>6.635</td>\n",
       "      <td>0.490</td>\n",
       "      <td>53.400</td>\n",
       "      <td>0.626</td>\n",
       "      <td>-0.024</td>\n",
       "      <td>0.607</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>140</th>\n",
       "      <td>Yemen</td>\n",
       "      <td>Middle East and North Africa</td>\n",
       "      <td>3.658</td>\n",
       "      <td>7.578</td>\n",
       "      <td>0.832</td>\n",
       "      <td>57.122</td>\n",
       "      <td>0.602</td>\n",
       "      <td>-0.147</td>\n",
       "      <td>0.800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>141</th>\n",
       "      <td>Tanzania</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.623</td>\n",
       "      <td>7.876</td>\n",
       "      <td>0.702</td>\n",
       "      <td>57.999</td>\n",
       "      <td>0.833</td>\n",
       "      <td>0.183</td>\n",
       "      <td>0.577</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>142</th>\n",
       "      <td>Haiti</td>\n",
       "      <td>Latin America and Caribbean</td>\n",
       "      <td>3.615</td>\n",
       "      <td>7.477</td>\n",
       "      <td>0.540</td>\n",
       "      <td>55.700</td>\n",
       "      <td>0.593</td>\n",
       "      <td>0.422</td>\n",
       "      <td>0.721</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>143</th>\n",
       "      <td>Malawi</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.600</td>\n",
       "      <td>6.958</td>\n",
       "      <td>0.537</td>\n",
       "      <td>57.948</td>\n",
       "      <td>0.780</td>\n",
       "      <td>0.038</td>\n",
       "      <td>0.729</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>144</th>\n",
       "      <td>Lesotho</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.512</td>\n",
       "      <td>7.926</td>\n",
       "      <td>0.787</td>\n",
       "      <td>48.700</td>\n",
       "      <td>0.715</td>\n",
       "      <td>-0.131</td>\n",
       "      <td>0.915</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>Botswana</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.467</td>\n",
       "      <td>9.782</td>\n",
       "      <td>0.784</td>\n",
       "      <td>59.269</td>\n",
       "      <td>0.824</td>\n",
       "      <td>-0.246</td>\n",
       "      <td>0.801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>Rwanda</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.415</td>\n",
       "      <td>7.676</td>\n",
       "      <td>0.552</td>\n",
       "      <td>61.400</td>\n",
       "      <td>0.897</td>\n",
       "      <td>0.061</td>\n",
       "      <td>0.167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>3.145</td>\n",
       "      <td>7.943</td>\n",
       "      <td>0.750</td>\n",
       "      <td>56.201</td>\n",
       "      <td>0.677</td>\n",
       "      <td>-0.047</td>\n",
       "      <td>0.821</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>South Asia</td>\n",
       "      <td>2.523</td>\n",
       "      <td>7.695</td>\n",
       "      <td>0.463</td>\n",
       "      <td>52.493</td>\n",
       "      <td>0.382</td>\n",
       "      <td>-0.102</td>\n",
       "      <td>0.924</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Country name            Regional indicator  Ladder score  \\\n",
       "139      Burundi            Sub-Saharan Africa         3.775   \n",
       "140        Yemen  Middle East and North Africa         3.658   \n",
       "141     Tanzania            Sub-Saharan Africa         3.623   \n",
       "142        Haiti   Latin America and Caribbean         3.615   \n",
       "143       Malawi            Sub-Saharan Africa         3.600   \n",
       "144      Lesotho            Sub-Saharan Africa         3.512   \n",
       "145     Botswana            Sub-Saharan Africa         3.467   \n",
       "146       Rwanda            Sub-Saharan Africa         3.415   \n",
       "147     Zimbabwe            Sub-Saharan Africa         3.145   \n",
       "148  Afghanistan                    South Asia         2.523   \n",
       "\n",
       "     Logged GDP per capita  Social support  Healthy life expectancy  \\\n",
       "139                  6.635           0.490                   53.400   \n",
       "140                  7.578           0.832                   57.122   \n",
       "141                  7.876           0.702                   57.999   \n",
       "142                  7.477           0.540                   55.700   \n",
       "143                  6.958           0.537                   57.948   \n",
       "144                  7.926           0.787                   48.700   \n",
       "145                  9.782           0.784                   59.269   \n",
       "146                  7.676           0.552                   61.400   \n",
       "147                  7.943           0.750                   56.201   \n",
       "148                  7.695           0.463                   52.493   \n",
       "\n",
       "     Freedom to make life choices  Generosity  Perceptions of corruption  \n",
       "139                         0.626      -0.024                      0.607  \n",
       "140                         0.602      -0.147                      0.800  \n",
       "141                         0.833       0.183                      0.577  \n",
       "142                         0.593       0.422                      0.721  \n",
       "143                         0.780       0.038                      0.729  \n",
       "144                         0.715      -0.131                      0.915  \n",
       "145                         0.824      -0.246                      0.801  \n",
       "146                         0.897       0.061                      0.167  \n",
       "147                         0.677      -0.047                      0.821  \n",
       "148                         0.382      -0.102                      0.924  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Top 10 of saddest country\n",
    "bottom_10"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97ada543",
   "metadata": {},
   "source": [
    "## Healthy life expectancy in happiest countries and saddiest countries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "da9b4eb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/3693299084.py:7: UserWarning: FixedFormatter should only be used together with FixedLocator\n",
      "  ax[0].set_xticklabels(top,rotation=45,ha='right')\n",
      "/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/3693299084.py:18: UserWarning: FixedFormatter should only be used together with FixedLocator\n",
      "  ax[1].set_xticklabels(bottom,rotation=45,ha='right')\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(1, 2, figsize=(16, 6))\n",
    "\n",
    "top=top_10['Country name']\n",
    "\n",
    "fig.suptitle(' life expectancy ',fontsize = 16)\n",
    "\n",
    "ax[0].set_xticklabels(top,rotation=45,ha='right')\n",
    "ax[0].set_title('life expectancy of happiest countries ')\n",
    "\n",
    "sns.barplot(x=top_10['Country name'],y=top_10['Healthy life expectancy'],ax=ax[0])\n",
    "ax[0].set_xlabel('Country name')\n",
    "ax[0].set_ylabel('life expectancy')\n",
    "\n",
    "bottom=bottom_10['Country name']\n",
    "\n",
    "\n",
    "\n",
    "ax[1].set_xticklabels(bottom,rotation=45,ha='right')\n",
    "ax[1].set_title('life expectancy of saddiest countries ')\n",
    "\n",
    "sns.barplot(x=bottom_10['Country name'],y=bottom_10['Healthy life expectancy'],ax=ax[1])\n",
    "ax[1].set_xlabel('Country name')\n",
    "ax[1].set_ylabel('life expectancy');\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c6a72fc",
   "metadata": {},
   "source": [
    "# Models"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a7db4124",
   "metadata": {},
   "source": [
    "# KNN Regressor Algorithm Implementation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "d9048b7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "x=data2021.drop([\"Ladder score\"],axis=1)\n",
    "y=data2021[[\"Ladder score\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "c64fd47a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder \n",
    "le = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "0fe4411c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1=le.fit_transform(x[\"Country name\"])\n",
    "x[\"Country name\"]= df1\n",
    "df1=le.fit_transform(x[\"Regional indicator\"])\n",
    "x[\"Regional indicator\"]= df1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc4579dc",
   "metadata": {},
   "source": [
    "### Preprocessing of data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "89687c34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>Regional indicator</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.790484</td>\n",
       "      <td>1.254062</td>\n",
       "      <td>1.162885</td>\n",
       "      <td>1.216171</td>\n",
       "      <td>1.039750</td>\n",
       "      <td>1.393550</td>\n",
       "      <td>-0.551886</td>\n",
       "      <td>-3.031228</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.953231</td>\n",
       "      <td>1.254062</td>\n",
       "      <td>1.299717</td>\n",
       "      <td>1.216171</td>\n",
       "      <td>1.143618</td>\n",
       "      <td>1.366990</td>\n",
       "      <td>0.300594</td>\n",
       "      <td>-3.070416</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.255474</td>\n",
       "      <td>1.254062</td>\n",
       "      <td>1.459064</td>\n",
       "      <td>1.111370</td>\n",
       "      <td>1.395869</td>\n",
       "      <td>1.127948</td>\n",
       "      <td>0.267294</td>\n",
       "      <td>-2.437802</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.464991</td>\n",
       "      <td>1.254062</td>\n",
       "      <td>1.252086</td>\n",
       "      <td>1.469440</td>\n",
       "      <td>1.188133</td>\n",
       "      <td>1.446671</td>\n",
       "      <td>1.166393</td>\n",
       "      <td>-0.304829</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.511490</td>\n",
       "      <td>1.254062</td>\n",
       "      <td>1.298851</td>\n",
       "      <td>1.111370</td>\n",
       "      <td>1.099103</td>\n",
       "      <td>1.074828</td>\n",
       "      <td>1.266293</td>\n",
       "      <td>-2.180278</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Country name  Regional indicator  Logged GDP per capita  Social support  \\\n",
       "0     -0.790484            1.254062               1.162885        1.216171   \n",
       "1     -0.953231            1.254062               1.299717        1.216171   \n",
       "2      1.255474            1.254062               1.459064        1.111370   \n",
       "3     -0.464991            1.254062               1.252086        1.469440   \n",
       "4      0.511490            1.254062               1.298851        1.111370   \n",
       "\n",
       "   Healthy life expectancy  Freedom to make life choices  Generosity  \\\n",
       "0                 1.039750                      1.393550   -0.551886   \n",
       "1                 1.143618                      1.366990    0.300594   \n",
       "2                 1.395869                      1.127948    0.267294   \n",
       "3                 1.188133                      1.446671    1.166393   \n",
       "4                 1.099103                      1.074828    1.266293   \n",
       "\n",
       "   Perceptions of corruption  \n",
       "0                  -3.031228  \n",
       "1                  -3.070416  \n",
       "2                  -2.437802  \n",
       "3                  -0.304829  \n",
       "4                  -2.180278  "
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler1=StandardScaler()\n",
    "X=pd.DataFrame(scaler1.fit_transform(x),columns=x.columns)\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "058788f7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Ladder score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.842</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.620</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.554</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.464</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Ladder score\n",
       "0         7.842\n",
       "1         7.620\n",
       "2         7.571\n",
       "3         7.554\n",
       "4         7.464"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f62aa201",
   "metadata": {},
   "source": [
    "## Train_Test_split "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "2add501b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(119, 8)\n",
      "(30, 8)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=20)\n",
    "print(X_train.shape)\n",
    "print(X_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bb959c1",
   "metadata": {},
   "source": [
    "## KNN Regressor "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "99871557",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "KNR = KNeighborsRegressor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "2fa1f865",
   "metadata": {},
   "outputs": [],
   "source": [
    "knnreg = KNR.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "a0f5320d",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = knnreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "7d9b07f1",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'str' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/3578839455.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mpredictions\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness score vs predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'str' object is not callable"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(y_test,predictions)\n",
    "plt.title(\"Actual happiness score vs predicted happiness scores\")\n",
    "plt.xlabel(\"Actual happiness scores\")\n",
    "plt.ylabel(\"Predicted happiness scores\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "106dd01c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#checking the errors and r2 error\n",
    "MAE_KNN= mean_absolute_error(y_test, predictions)\n",
    "MSE_KNN= mean_squared_error(y_test, predictions)\n",
    "R2_KNN = r2_score(y_test, predictions)\n",
    "\n",
    "print(\"THe MAE of the KNN_regression is:\", MAE_KNN)\n",
    "print(\"THe MSE of the KNN_regression is:\", MSE_KNN)\n",
    "print(\"THe R^2 of the KNN_regression is:\", R2_KNN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "5943544a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Mean Absolute Error</th>\n",
       "      <th>Mean Squared Error</th>\n",
       "      <th>R Squared Error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KNN_REGRESSOR</td>\n",
       "      <td>0.39828</td>\n",
       "      <td>0.341891</td>\n",
       "      <td>0.677368</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Model  Mean Absolute Error  Mean Squared Error  R Squared Error\n",
       "0  KNN_REGRESSOR              0.39828            0.341891         0.677368"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_table = pd.DataFrame(columns=[\"Model\",\"Mean Absolute Error\",\"Mean Squared Error\",\"R Squared Error\"])\n",
    "\n",
    "dict1 = {\n",
    "\"Model\" : \"KNN_REGRESSOR\",\n",
    "\"Mean Absolute Error\" : mean_absolute_error(y_test, predictions),\n",
    "\"Mean Squared Error\" :mean_squared_error(y_test, predictions),\n",
    "\"R Squared Error\" : r2_score(y_test, predictions),\n",
    "}\n",
    "df_table = df_table.append(dict1,ignore_index = True)\n",
    "df_table"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "683c3977",
   "metadata": {},
   "source": [
    "## Linear Regression "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "5adfab6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Linear Regressor\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lin_reg = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "eb460935",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lin_reg.fit(X_train,y_train)\n",
    "#print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "f047f9da",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_1= lin_reg.predict(X_test)\n",
    "#lin_reg.predict([[43,1.340,1.587,0.986,0.596,0.393]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "0f65c31b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The intercept for the model is 5.550604570952311\n"
     ]
    }
   ],
   "source": [
    "#finding the intercept\n",
    "intercept = lin_reg.intercept_[0]\n",
    "print(\"The intercept for the model is {}\".format(intercept))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "f89d0e99",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "THe MAE of the Linear_regression is: 0.38040182564004554\n",
      "THe MSE of the Linear_regression is: 0.2401828467207272\n",
      "THe R^2 of the Linear_regression is: 0.7733466507728999\n"
     ]
    }
   ],
   "source": [
    "#checking the errors and r2 error\n",
    "\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import r2_score\n",
    "\n",
    "MAE_LR= mean_absolute_error(y_test, predictions_1)\n",
    "MSE_LR= mean_squared_error(y_test, predictions_1)\n",
    "R2_LR = r2_score(y_test, predictions_1)\n",
    "\n",
    "print(\"THe MAE of the Linear_regression is:\", MAE_LR)\n",
    "print(\"THe MSE of the Linear_regression is:\", MSE_LR)\n",
    "print(\"THe R^2 of the Linear_regression is:\", R2_LR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "1aee2d0c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Mean Absolute Error</th>\n",
       "      <th>Mean Squared Error</th>\n",
       "      <th>R Squared Error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KNN_REGRESSOR</td>\n",
       "      <td>0.398280</td>\n",
       "      <td>0.341891</td>\n",
       "      <td>0.677368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LINEAR_REGRESSOR</td>\n",
       "      <td>0.380402</td>\n",
       "      <td>0.240183</td>\n",
       "      <td>0.773347</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Model  Mean Absolute Error  Mean Squared Error  R Squared Error\n",
       "0     KNN_REGRESSOR             0.398280            0.341891         0.677368\n",
       "1  LINEAR_REGRESSOR             0.380402            0.240183         0.773347"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dict2= {\"Model\" : \"LINEAR_REGRESSOR\",\n",
    "    \"Mean Absolute Error\": mean_absolute_error(y_test, predictions_1),\n",
    "\"Mean Squared Error\":mean_squared_error(y_test, predictions_1),\n",
    "\"R Squared Error\" : r2_score(y_test, predictions_1)\n",
    "}\n",
    "df_table = df_table.append(dict2,ignore_index=True)\n",
    "df_table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "3d4cb287",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'str' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/394108808.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mpredictions_1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness score vs predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'str' object is not callable"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Vizualization\n",
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(y_test,predictions_1)\n",
    "plt.title(\"Actual happiness score vs predicted happiness scores\")\n",
    "plt.xlabel(\"Actual happiness scores\")\n",
    "plt.ylabel(\"Predicted happiness scores\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c75b7df",
   "metadata": {},
   "source": [
    "# Support Vector Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "9d9aef07",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVR()"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.svm import SVR\n",
    "regressor=SVR(kernel=\"rbf\")\n",
    "regressor.fit(X_train,np.array(y_train).ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "ca8004ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_2=regressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "56ef640e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "THe MAE of the Support_Vector_Regressor is: 0.41502675631552466\n",
      "THe MSE of the Support_Vector_Regressor is: 0.3815743250334292\n",
      "THe R^2 of the Support_Vector_Regressor is: 0.639919753101863\n"
     ]
    }
   ],
   "source": [
    "MAE_SVR = mean_absolute_error(y_test, predictions_2)\n",
    "MSE_SVR = mean_squared_error(y_test, predictions_2)\n",
    "R2_SVR = r2_score(y_test, predictions_2)\n",
    "print(\"THe MAE of the Support_Vector_Regressor is:\", MAE_SVR)\n",
    "print(\"THe MSE of the Support_Vector_Regressor is:\", MSE_SVR)\n",
    "print(\"THe R^2 of the Support_Vector_Regressor is:\", R2_SVR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "4eab5daf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Mean Absolute Error</th>\n",
       "      <th>Mean Squared Error</th>\n",
       "      <th>R Squared Error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KNN_REGRESSOR</td>\n",
       "      <td>0.398280</td>\n",
       "      <td>0.341891</td>\n",
       "      <td>0.677368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LINEAR_REGRESSOR</td>\n",
       "      <td>0.380402</td>\n",
       "      <td>0.240183</td>\n",
       "      <td>0.773347</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SUPPORT_VECTOR_REGRESSOR</td>\n",
       "      <td>0.415027</td>\n",
       "      <td>0.381574</td>\n",
       "      <td>0.639920</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      Model  Mean Absolute Error  Mean Squared Error  \\\n",
       "0             KNN_REGRESSOR             0.398280            0.341891   \n",
       "1          LINEAR_REGRESSOR             0.380402            0.240183   \n",
       "2  SUPPORT_VECTOR_REGRESSOR             0.415027            0.381574   \n",
       "\n",
       "   R Squared Error  \n",
       "0         0.677368  \n",
       "1         0.773347  \n",
       "2         0.639920  "
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dict3= {\"Model\" : \"SUPPORT_VECTOR_REGRESSOR\",\n",
    "    \"Mean Absolute Error\": mean_absolute_error(y_test, predictions_2),\n",
    "\"Mean Squared Error\":mean_squared_error(y_test, predictions_2),\n",
    "\"R Squared Error\" : r2_score(y_test, predictions_2)\n",
    "}\n",
    "df_table = df_table.append(dict3,ignore_index=True)\n",
    "df_table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "b4c908d1",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'str' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/431167484.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mpredictions_2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness score vs predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'str' object is not callable"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(y_test,predictions_2)\n",
    "plt.title(\"Actual happiness score vs predicted happiness scores\")\n",
    "plt.xlabel(\"Actual happiness scores\")\n",
    "plt.ylabel(\"Predicted happiness scores\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "048f88df",
   "metadata": {},
   "source": [
    "# Decision Tree Regressor "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "7936d6dd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeRegressor()"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#decision Tree Regressor\n",
    "\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "regressor=DecisionTreeRegressor()\n",
    "regressor.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "10c287b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_3=regressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "b82c28ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "THe MAE of the Decision_Tree is: 0.6723333333333333\n",
      "THe MSE of the Decision_Tree is: 0.6530973333333333\n",
      "THe R^2 of the Decision_Tree is: 0.38369163330216505\n"
     ]
    }
   ],
   "source": [
    "MAE_DTR = mean_absolute_error(y_test, predictions_3)\n",
    "MSE_DTR = mean_squared_error(y_test, predictions_3)\n",
    "R2_DTR = r2_score(y_test, predictions_3)\n",
    "print(\"THe MAE of the Decision_Tree is:\", MAE_DTR)\n",
    "print(\"THe MSE of the Decision_Tree is:\", MSE_DTR)\n",
    "print(\"THe R^2 of the Decision_Tree is:\", R2_DTR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "01dddadd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Mean Absolute Error</th>\n",
       "      <th>Mean Squared Error</th>\n",
       "      <th>R Squared Error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KNN_REGRESSOR</td>\n",
       "      <td>0.398280</td>\n",
       "      <td>0.341891</td>\n",
       "      <td>0.677368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LINEAR_REGRESSOR</td>\n",
       "      <td>0.380402</td>\n",
       "      <td>0.240183</td>\n",
       "      <td>0.773347</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SUPPORT_VECTOR_REGRESSOR</td>\n",
       "      <td>0.415027</td>\n",
       "      <td>0.381574</td>\n",
       "      <td>0.639920</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>DECISION_TREE_REGRESSOR</td>\n",
       "      <td>0.672333</td>\n",
       "      <td>0.653097</td>\n",
       "      <td>0.383692</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      Model  Mean Absolute Error  Mean Squared Error  \\\n",
       "0             KNN_REGRESSOR             0.398280            0.341891   \n",
       "1          LINEAR_REGRESSOR             0.380402            0.240183   \n",
       "2  SUPPORT_VECTOR_REGRESSOR             0.415027            0.381574   \n",
       "3   DECISION_TREE_REGRESSOR             0.672333            0.653097   \n",
       "\n",
       "   R Squared Error  \n",
       "0         0.677368  \n",
       "1         0.773347  \n",
       "2         0.639920  \n",
       "3         0.383692  "
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dict4= {\"Model\" : \"DECISION_TREE_REGRESSOR\",\n",
    "    \"Mean Absolute Error\": mean_absolute_error(y_test, predictions_3),\n",
    "\"Mean Squared Error\":mean_squared_error(y_test, predictions_3),\n",
    "\"R Squared Error\" : r2_score(y_test, predictions_3)\n",
    "}\n",
    "df_table = df_table.append(dict4,ignore_index=True)\n",
    "df_table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "061ea9c4",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'str' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/1305742492.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mpredictions_3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actaul happiness score vs predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'str' object is not callable"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(y_test,predictions_3)\n",
    "plt.title(\"Actaul happiness score vs predicted happiness scores\")\n",
    "plt.xlabel(\"Actual happiness scores\")\n",
    "plt.ylabel(\"Predicted happiness scores\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b16263da",
   "metadata": {},
   "source": [
    "# Random Forest Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "da6f2b08",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "params_to_test = {\n",
    "    'n_estimators':range(1,11)\n",
    "}\n",
    "\n",
    "rf_model=RandomForestRegressor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "8a80fdeb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(estimator=RandomForestRegressor(),\n",
       "             param_grid={'n_estimators': range(1, 11)})"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_search=GridSearchCV(rf_model, param_grid=params_to_test)\n",
    "\n",
    "grid_search.fit(X_train,np.array(y_train).ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "43eba044",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_4=grid_search.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "0a6e8cce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "THe MAE of the Random_forest is: 0.44400666666666655\n",
      "THe MSE of the Random_forest is: 0.3192767906666666\n",
      "THe R^2 of the Random_forest is: 0.6987080679444924\n"
     ]
    }
   ],
   "source": [
    "MAE_RF = mean_absolute_error(y_test, predictions_4)\n",
    "MSE_RF = mean_squared_error(y_test, predictions_4)\n",
    "R2_RF = r2_score(y_test, predictions_4)\n",
    "print(\"THe MAE of the Random_forest is:\",MAE_RF)\n",
    "print(\"THe MSE of the Random_forest is:\", MSE_RF)\n",
    "print(\"THe R^2 of the Random_forest is:\", R2_RF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "1deaed88",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'str' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/1517099451.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mpredictions_4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness score vs predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'str' object is not callable"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(y_test,predictions_4)\n",
    "plt.title(\"Actual happiness score vs predicted happiness scores\")\n",
    "plt.xlabel(\"Actual happiness scores\")\n",
    "plt.ylabel(\"Predicted happiness scores\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "ff9d1b9b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Model</th>\n",
       "      <th>Mean Absolute Error</th>\n",
       "      <th>Mean Squared Error</th>\n",
       "      <th>R Squared Error</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KNN_REGRESSOR</td>\n",
       "      <td>0.398280</td>\n",
       "      <td>0.341891</td>\n",
       "      <td>0.677368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LINEAR_REGRESSOR</td>\n",
       "      <td>0.380402</td>\n",
       "      <td>0.240183</td>\n",
       "      <td>0.773347</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SUPPORT_VECTOR_REGRESSOR</td>\n",
       "      <td>0.415027</td>\n",
       "      <td>0.381574</td>\n",
       "      <td>0.639920</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>DECISION_TREE_REGRESSOR</td>\n",
       "      <td>0.672333</td>\n",
       "      <td>0.653097</td>\n",
       "      <td>0.383692</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>RANDOM_FOREST_REGRESSOR</td>\n",
       "      <td>0.444007</td>\n",
       "      <td>0.319277</td>\n",
       "      <td>0.698708</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      Model  Mean Absolute Error  Mean Squared Error  \\\n",
       "0             KNN_REGRESSOR             0.398280            0.341891   \n",
       "1          LINEAR_REGRESSOR             0.380402            0.240183   \n",
       "2  SUPPORT_VECTOR_REGRESSOR             0.415027            0.381574   \n",
       "3   DECISION_TREE_REGRESSOR             0.672333            0.653097   \n",
       "4   RANDOM_FOREST_REGRESSOR             0.444007            0.319277   \n",
       "\n",
       "   R Squared Error  \n",
       "0         0.677368  \n",
       "1         0.773347  \n",
       "2         0.639920  \n",
       "3         0.383692  \n",
       "4         0.698708  "
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dict5= {\"Model\" : \"RANDOM_FOREST_REGRESSOR\",\n",
    "    \"Mean Absolute Error\": mean_absolute_error(y_test, predictions_4),\n",
    "\"Mean Squared Error\":mean_squared_error(y_test, predictions_4),\n",
    "\"R Squared Error\" : r2_score(y_test, predictions_4)\n",
    "}\n",
    "df_table = df_table.append(dict5,ignore_index=True)\n",
    "df_table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "e98792cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: xgboost in /opt/anaconda3/lib/python3.9/site-packages (1.5.0)\n",
      "Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.9/site-packages (from xgboost) (1.19.5)\n",
      "Requirement already satisfied: scipy in /opt/anaconda3/lib/python3.9/site-packages (from xgboost) (1.7.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install xgboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "07f19f87",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,\n",
       "             gamma=0, gpu_id=-1, importance_type=None,\n",
       "             interaction_constraints='', learning_rate=0.300000012,\n",
       "             max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,\n",
       "             monotone_constraints='()', n_estimators=100, n_jobs=8,\n",
       "             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,\n",
       "             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',\n",
       "             validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from xgboost import XGBRegressor\n",
    "xgreg = XGBRegressor()\n",
    "xgreg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "e9df3847",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_5 = xgreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "db81ec64",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "THe MAE of the XG Boost Regressor is: 0.33036000518798825\n",
      "THe MSE of the XG Boost Regressor  is: 0.20255680351365604\n",
      "THe R^2 of the XG Boost Regressor  is: 0.8088532193204959\n"
     ]
    }
   ],
   "source": [
    "MAE_RF = mean_absolute_error(y_test, predictions_5)\n",
    "MSE_RF = mean_squared_error(y_test, predictions_5)\n",
    "R2_RF = r2_score(y_test, predictions_5)\n",
    "print(\"THe MAE of the XG Boost Regressor is:\",MAE_RF)\n",
    "print(\"THe MSE of the XG Boost Regressor  is:\", MSE_RF)\n",
    "print(\"THe R^2 of the XG Boost Regressor  is:\", R2_RF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "f5287e1d",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'str' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/9t/64tm8pcn4vjg_vv49fm471rw0000gn/T/ipykernel_24681/3490837739.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfigure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscatter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mpredictions_5\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness score vs predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Actual happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Predicted happiness scores\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: 'str' object is not callable"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(y_test,predictions_5)\n",
    "plt.title(\"Actual happiness score vs predicted happiness scores\")\n",
    "plt.xlabel(\"Actual happiness scores\")\n",
    "plt.ylabel(\"Predicted happiness scores\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d327dd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "dict6= {\"Model\" : \"XG_BOOST_REGRESSOR\",\n",
    "    \"Mean Absolute Error\": mean_absolute_error(y_test, predictions_5),\n",
    "\"Mean Squared Error\":mean_squared_error(y_test, predictions_5),\n",
    "\"R Squared Error\" : r2_score(y_test, predictions_5)\n",
    "}\n",
    "df_table = df_table.append(dict6,ignore_index=True)\n",
    "df_table"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71e95ffd",
   "metadata": {},
   "source": [
    "# Conclusion\n",
    "XGBoost regressor is giving the best results"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b40e649e",
   "metadata": {},
   "source": [
    "# merge two datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "74b559ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop a column we don't need\n",
    "data2021 = data2021.drop(['Regional indicator'],axis=1)\n",
    "#Add year feature for data2021\n",
    "data2021['year']=2021\n",
    "new_data=pd.merge(data,data2021,how='outer')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2f43bacf",
   "metadata": {},
   "source": [
    "### review and cleaning new data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "efca9eff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Country name</th>\n",
       "      <th>year</th>\n",
       "      <th>Ladder score</th>\n",
       "      <th>Logged GDP per capita</th>\n",
       "      <th>Social support</th>\n",
       "      <th>Healthy life expectancy</th>\n",
       "      <th>Freedom to make life choices</th>\n",
       "      <th>Generosity</th>\n",
       "      <th>Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2008</td>\n",
       "      <td>3.724</td>\n",
       "      <td>7.370</td>\n",
       "      <td>0.451</td>\n",
       "      <td>50.800</td>\n",
       "      <td>0.718</td>\n",
       "      <td>0.168</td>\n",
       "      <td>0.882</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2009</td>\n",
       "      <td>4.402</td>\n",
       "      <td>7.540</td>\n",
       "      <td>0.552</td>\n",
       "      <td>51.200</td>\n",
       "      <td>0.679</td>\n",
       "      <td>0.190</td>\n",
       "      <td>0.850</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2010</td>\n",
       "      <td>4.758</td>\n",
       "      <td>7.647</td>\n",
       "      <td>0.539</td>\n",
       "      <td>51.600</td>\n",
       "      <td>0.600</td>\n",
       "      <td>0.121</td>\n",
       "      <td>0.707</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2011</td>\n",
       "      <td>3.832</td>\n",
       "      <td>7.620</td>\n",
       "      <td>0.521</td>\n",
       "      <td>51.920</td>\n",
       "      <td>0.496</td>\n",
       "      <td>0.162</td>\n",
       "      <td>0.731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2012</td>\n",
       "      <td>3.783</td>\n",
       "      <td>7.705</td>\n",
       "      <td>0.521</td>\n",
       "      <td>52.240</td>\n",
       "      <td>0.531</td>\n",
       "      <td>0.236</td>\n",
       "      <td>0.776</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1852</th>\n",
       "      <td>Lesotho</td>\n",
       "      <td>2021</td>\n",
       "      <td>3.512</td>\n",
       "      <td>7.926</td>\n",
       "      <td>0.787</td>\n",
       "      <td>48.700</td>\n",
       "      <td>0.715</td>\n",
       "      <td>-0.131</td>\n",
       "      <td>0.915</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1853</th>\n",
       "      <td>Botswana</td>\n",
       "      <td>2021</td>\n",
       "      <td>3.467</td>\n",
       "      <td>9.782</td>\n",
       "      <td>0.784</td>\n",
       "      <td>59.269</td>\n",
       "      <td>0.824</td>\n",
       "      <td>-0.246</td>\n",
       "      <td>0.801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1854</th>\n",
       "      <td>Rwanda</td>\n",
       "      <td>2021</td>\n",
       "      <td>3.415</td>\n",
       "      <td>7.676</td>\n",
       "      <td>0.552</td>\n",
       "      <td>61.400</td>\n",
       "      <td>0.897</td>\n",
       "      <td>0.061</td>\n",
       "      <td>0.167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1855</th>\n",
       "      <td>Zimbabwe</td>\n",
       "      <td>2021</td>\n",
       "      <td>3.145</td>\n",
       "      <td>7.943</td>\n",
       "      <td>0.750</td>\n",
       "      <td>56.201</td>\n",
       "      <td>0.677</td>\n",
       "      <td>-0.047</td>\n",
       "      <td>0.821</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1856</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>2021</td>\n",
       "      <td>2.523</td>\n",
       "      <td>7.695</td>\n",
       "      <td>0.463</td>\n",
       "      <td>52.493</td>\n",
       "      <td>0.382</td>\n",
       "      <td>-0.102</td>\n",
       "      <td>0.924</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1857 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Country name  year  Ladder score  Logged GDP per capita  Social support  \\\n",
       "0     Afghanistan  2008         3.724                  7.370           0.451   \n",
       "1     Afghanistan  2009         4.402                  7.540           0.552   \n",
       "2     Afghanistan  2010         4.758                  7.647           0.539   \n",
       "3     Afghanistan  2011         3.832                  7.620           0.521   \n",
       "4     Afghanistan  2012         3.783                  7.705           0.521   \n",
       "...           ...   ...           ...                    ...             ...   \n",
       "1852      Lesotho  2021         3.512                  7.926           0.787   \n",
       "1853     Botswana  2021         3.467                  9.782           0.784   \n",
       "1854       Rwanda  2021         3.415                  7.676           0.552   \n",
       "1855     Zimbabwe  2021         3.145                  7.943           0.750   \n",
       "1856  Afghanistan  2021         2.523                  7.695           0.463   \n",
       "\n",
       "      Healthy life expectancy  Freedom to make life choices  Generosity  \\\n",
       "0                      50.800                         0.718       0.168   \n",
       "1                      51.200                         0.679       0.190   \n",
       "2                      51.600                         0.600       0.121   \n",
       "3                      51.920                         0.496       0.162   \n",
       "4                      52.240                         0.531       0.236   \n",
       "...                       ...                           ...         ...   \n",
       "1852                   48.700                         0.715      -0.131   \n",
       "1853                   59.269                         0.824      -0.246   \n",
       "1854                   61.400                         0.897       0.061   \n",
       "1855                   56.201                         0.677      -0.047   \n",
       "1856                   52.493                         0.382      -0.102   \n",
       "\n",
       "      Perceptions of corruption  \n",
       "0                         0.882  \n",
       "1                         0.850  \n",
       "2                         0.707  \n",
       "3                         0.731  \n",
       "4                         0.776  \n",
       "...                         ...  \n",
       "1852                      0.915  \n",
       "1853                      0.801  \n",
       "1854                      0.167  \n",
       "1855                      0.821  \n",
       "1856                      0.924  \n",
       "\n",
       "[1857 rows x 9 columns]"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "a2116b3e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1857, 9)"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#size of new data\n",
    "new_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "f64134b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 1857 entries, 0 to 1856\n",
      "Data columns (total 9 columns):\n",
      " #   Column                        Non-Null Count  Dtype  \n",
      "---  ------                        --------------  -----  \n",
      " 0   Country name                  1857 non-null   object \n",
      " 1   year                          1857 non-null   int64  \n",
      " 2   Ladder score                  1857 non-null   float64\n",
      " 3   Logged GDP per capita         1857 non-null   float64\n",
      " 4   Social support                1857 non-null   float64\n",
      " 5   Healthy life expectancy       1857 non-null   float64\n",
      " 6   Freedom to make life choices  1857 non-null   float64\n",
      " 7   Generosity                    1857 non-null   float64\n",
      " 8   Perceptions of corruption     1857 non-null   float64\n",
      "dtypes: float64(7), int64(1), object(1)\n",
      "memory usage: 145.1+ KB\n"
     ]
    }
   ],
   "source": [
    "# Information about the Variables\n",
    "new_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "5a0b1719",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Country name                    0\n",
       "year                            0\n",
       "Ladder score                    0\n",
       "Logged GDP per capita           0\n",
       "Social support                  0\n",
       "Healthy life expectancy         0\n",
       "Freedom to make life choices    0\n",
       "Generosity                      0\n",
       "Perceptions of corruption       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#missing data\n",
    "new_data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "ed533b6c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "#T_620b4_row0_col0, #T_620b4_row1_col1, #T_620b4_row2_col2, #T_620b4_row3_col3, #T_620b4_row4_col4, #T_620b4_row5_col5, #T_620b4_row6_col6, #T_620b4_row7_col7 {\n",
       "  background-color: #b40426;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row0_col1 {\n",
       "  background-color: #b1cbfc;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row0_col2 {\n",
       "  background-color: #a6c4fe;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row0_col3 {\n",
       "  background-color: #799cf8;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row0_col4 {\n",
       "  background-color: #bed2f6;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row0_col5 {\n",
       "  background-color: #dddcdc;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row0_col6 {\n",
       "  background-color: #7699f6;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row0_col7, #T_620b4_row4_col0 {\n",
       "  background-color: #90b2fe;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row1_col0 {\n",
       "  background-color: #6a8bef;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row1_col2 {\n",
       "  background-color: #e46e56;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row1_col3 {\n",
       "  background-color: #f29274;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row1_col4 {\n",
       "  background-color: #ea7b60;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row1_col5 {\n",
       "  background-color: #f7b396;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row1_col6 {\n",
       "  background-color: #afcafc;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row1_col7 {\n",
       "  background-color: #4257c9;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row2_col0, #T_620b4_row3_col7 {\n",
       "  background-color: #7295f4;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row2_col1 {\n",
       "  background-color: #e26952;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row2_col3 {\n",
       "  background-color: #f39475;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row2_col4, #T_620b4_row4_col2 {\n",
       "  background-color: #d75445;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row2_col5 {\n",
       "  background-color: #edd2c3;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row2_col6 {\n",
       "  background-color: #779af7;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row2_col7 {\n",
       "  background-color: #5673e0;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row3_col0 {\n",
       "  background-color: #5a78e4;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row3_col1 {\n",
       "  background-color: #ed8366;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row3_col2 {\n",
       "  background-color: #f08a6c;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row3_col4 {\n",
       "  background-color: #f6a586;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row3_col5 {\n",
       "  background-color: #f2cab5;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row3_col6 {\n",
       "  background-color: #8db0fe;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row4_col1 {\n",
       "  background-color: #e8765c;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row4_col3 {\n",
       "  background-color: #f7af91;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row4_col5 {\n",
       "  background-color: #f1cdba;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row4_col6, #T_620b4_row6_col3 {\n",
       "  background-color: #82a6fb;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row4_col7 {\n",
       "  background-color: #5875e1;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row5_col0 {\n",
       "  background-color: #a9c6fd;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row5_col1 {\n",
       "  background-color: #f7b599;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row5_col2, #T_620b4_row5_col3 {\n",
       "  background-color: #e2dad5;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row5_col4 {\n",
       "  background-color: #e8d6cc;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row5_col6 {\n",
       "  background-color: #d3dbe7;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row5_col7, #T_620b4_row7_col0, #T_620b4_row7_col1, #T_620b4_row7_col2, #T_620b4_row7_col3, #T_620b4_row7_col4, #T_620b4_row7_col5, #T_620b4_row7_col6 {\n",
       "  background-color: #3b4cc0;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row6_col0 {\n",
       "  background-color: #4a63d3;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row6_col1 {\n",
       "  background-color: #c7d7f0;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row6_col2 {\n",
       "  background-color: #85a8fc;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "#T_620b4_row6_col4 {\n",
       "  background-color: #8fb1fe;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row6_col5 {\n",
       "  background-color: #e6d7cf;\n",
       "  color: #000000;\n",
       "}\n",
       "#T_620b4_row6_col7 {\n",
       "  background-color: #6687ed;\n",
       "  color: #f1f1f1;\n",
       "}\n",
       "</style>\n",
       "<table id=\"T_620b4_\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th class=\"col_heading level0 col0\" >year</th>\n",
       "      <th class=\"col_heading level0 col1\" >Ladder score</th>\n",
       "      <th class=\"col_heading level0 col2\" >Logged GDP per capita</th>\n",
       "      <th class=\"col_heading level0 col3\" >Social support</th>\n",
       "      <th class=\"col_heading level0 col4\" >Healthy life expectancy</th>\n",
       "      <th class=\"col_heading level0 col5\" >Freedom to make life choices</th>\n",
       "      <th class=\"col_heading level0 col6\" >Generosity</th>\n",
       "      <th class=\"col_heading level0 col7\" >Perceptions of corruption</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row0\" class=\"row_heading level0 row0\" >year</th>\n",
       "      <td id=\"T_620b4_row0_col0\" class=\"data row0 col0\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row0_col1\" class=\"data row0 col1\" >0.064171</td>\n",
       "      <td id=\"T_620b4_row0_col2\" class=\"data row0 col2\" >0.090051</td>\n",
       "      <td id=\"T_620b4_row0_col3\" class=\"data row0 col3\" >0.013482</td>\n",
       "      <td id=\"T_620b4_row0_col4\" class=\"data row0 col4\" >0.185613</td>\n",
       "      <td id=\"T_620b4_row0_col5\" class=\"data row0 col5\" >0.264187</td>\n",
       "      <td id=\"T_620b4_row0_col6\" class=\"data row0 col6\" >-0.041746</td>\n",
       "      <td id=\"T_620b4_row0_col7\" class=\"data row0 col7\" >-0.100054</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row1\" class=\"row_heading level0 row1\" >Ladder score</th>\n",
       "      <td id=\"T_620b4_row1_col0\" class=\"data row1 col0\" >0.064171</td>\n",
       "      <td id=\"T_620b4_row1_col1\" class=\"data row1 col1\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row1_col2\" class=\"data row1 col2\" >0.792625</td>\n",
       "      <td id=\"T_620b4_row1_col3\" class=\"data row1 col3\" >0.716365</td>\n",
       "      <td id=\"T_620b4_row1_col4\" class=\"data row1 col4\" >0.755120</td>\n",
       "      <td id=\"T_620b4_row1_col5\" class=\"data row1 col5\" >0.528963</td>\n",
       "      <td id=\"T_620b4_row1_col6\" class=\"data row1 col6\" >0.167975</td>\n",
       "      <td id=\"T_620b4_row1_col7\" class=\"data row1 col7\" >-0.446527</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row2\" class=\"row_heading level0 row2\" >Logged GDP per capita</th>\n",
       "      <td id=\"T_620b4_row2_col0\" class=\"data row2 col0\" >0.090051</td>\n",
       "      <td id=\"T_620b4_row2_col1\" class=\"data row2 col1\" >0.792625</td>\n",
       "      <td id=\"T_620b4_row2_col2\" class=\"data row2 col2\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row2_col3\" class=\"data row2 col3\" >0.711879</td>\n",
       "      <td id=\"T_620b4_row2_col4\" class=\"data row2 col4\" >0.859453</td>\n",
       "      <td id=\"T_620b4_row2_col5\" class=\"data row2 col5\" >0.358373</td>\n",
       "      <td id=\"T_620b4_row2_col6\" class=\"data row2 col6\" >-0.038199</td>\n",
       "      <td id=\"T_620b4_row2_col7\" class=\"data row2 col7\" >-0.343938</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row3\" class=\"row_heading level0 row3\" >Social support</th>\n",
       "      <td id=\"T_620b4_row3_col0\" class=\"data row3 col0\" >0.013482</td>\n",
       "      <td id=\"T_620b4_row3_col1\" class=\"data row3 col1\" >0.716365</td>\n",
       "      <td id=\"T_620b4_row3_col2\" class=\"data row3 col2\" >0.711879</td>\n",
       "      <td id=\"T_620b4_row3_col3\" class=\"data row3 col3\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row3_col4\" class=\"data row3 col4\" >0.623813</td>\n",
       "      <td id=\"T_620b4_row3_col5\" class=\"data row3 col5\" >0.414582</td>\n",
       "      <td id=\"T_620b4_row3_col6\" class=\"data row3 col6\" >0.043773</td>\n",
       "      <td id=\"T_620b4_row3_col7\" class=\"data row3 col7\" >-0.225320</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row4\" class=\"row_heading level0 row4\" >Healthy life expectancy</th>\n",
       "      <td id=\"T_620b4_row4_col0\" class=\"data row4 col0\" >0.185613</td>\n",
       "      <td id=\"T_620b4_row4_col1\" class=\"data row4 col1\" >0.755120</td>\n",
       "      <td id=\"T_620b4_row4_col2\" class=\"data row4 col2\" >0.859453</td>\n",
       "      <td id=\"T_620b4_row4_col3\" class=\"data row4 col3\" >0.623813</td>\n",
       "      <td id=\"T_620b4_row4_col4\" class=\"data row4 col4\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row4_col5\" class=\"data row4 col5\" >0.392673</td>\n",
       "      <td id=\"T_620b4_row4_col6\" class=\"data row4 col6\" >0.004725</td>\n",
       "      <td id=\"T_620b4_row4_col7\" class=\"data row4 col7\" >-0.338215</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row5\" class=\"row_heading level0 row5\" >Freedom to make life choices</th>\n",
       "      <td id=\"T_620b4_row5_col0\" class=\"data row5 col0\" >0.264187</td>\n",
       "      <td id=\"T_620b4_row5_col1\" class=\"data row5 col1\" >0.528963</td>\n",
       "      <td id=\"T_620b4_row5_col2\" class=\"data row5 col2\" >0.358373</td>\n",
       "      <td id=\"T_620b4_row5_col3\" class=\"data row5 col3\" >0.414582</td>\n",
       "      <td id=\"T_620b4_row5_col4\" class=\"data row5 col4\" >0.392673</td>\n",
       "      <td id=\"T_620b4_row5_col5\" class=\"data row5 col5\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row5_col6\" class=\"data row5 col6\" >0.312524</td>\n",
       "      <td id=\"T_620b4_row5_col7\" class=\"data row5 col7\" >-0.482894</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row6\" class=\"row_heading level0 row6\" >Generosity</th>\n",
       "      <td id=\"T_620b4_row6_col0\" class=\"data row6 col0\" >-0.041746</td>\n",
       "      <td id=\"T_620b4_row6_col1\" class=\"data row6 col1\" >0.167975</td>\n",
       "      <td id=\"T_620b4_row6_col2\" class=\"data row6 col2\" >-0.038199</td>\n",
       "      <td id=\"T_620b4_row6_col3\" class=\"data row6 col3\" >0.043773</td>\n",
       "      <td id=\"T_620b4_row6_col4\" class=\"data row6 col4\" >0.004725</td>\n",
       "      <td id=\"T_620b4_row6_col5\" class=\"data row6 col5\" >0.312524</td>\n",
       "      <td id=\"T_620b4_row6_col6\" class=\"data row6 col6\" >1.000000</td>\n",
       "      <td id=\"T_620b4_row6_col7\" class=\"data row6 col7\" >-0.278394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_620b4_level0_row7\" class=\"row_heading level0 row7\" >Perceptions of corruption</th>\n",
       "      <td id=\"T_620b4_row7_col0\" class=\"data row7 col0\" >-0.100054</td>\n",
       "      <td id=\"T_620b4_row7_col1\" class=\"data row7 col1\" >-0.446527</td>\n",
       "      <td id=\"T_620b4_row7_col2\" class=\"data row7 col2\" >-0.343938</td>\n",
       "      <td id=\"T_620b4_row7_col3\" class=\"data row7 col3\" >-0.225320</td>\n",
       "      <td id=\"T_620b4_row7_col4\" class=\"data row7 col4\" >-0.338215</td>\n",
       "      <td id=\"T_620b4_row7_col5\" class=\"data row7 col5\" >-0.482894</td>\n",
       "      <td id=\"T_620b4_row7_col6\" class=\"data row7 col6\" >-0.278394</td>\n",
       "      <td id=\"T_620b4_row7_col7\" class=\"data row7 col7\" >1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x7fc33022cf70>"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#finding the correlation of the new_data\n",
    "corr=new_data.corr()\n",
    "corr.style.background_gradient(cmap='coolwarm')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
