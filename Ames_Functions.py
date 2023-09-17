import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, ElasticNet
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests
import json
import time
import geopy
import pickle
from geopy import Nominatim
from geopy.extra.rate_limiter import RateLimiter

locator = Nominatim(user_agent="myGeocoder")

# Replace YOUR_API_KEY with your actual API key. Sign up and get an API key on https://www.geoapify.com/ 
API_KEY = "c65731ef1fc94fd3aba9a53df17c2366"




def get_vif (df):
    return [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]



#Below code is from geoapify website
def get_geoapify (address):
    # Build the API URL
    url = f"https://api.geoapify.com/v1/geocode/search?text={address}&limit=1&apiKey={API_KEY}"

    # Send the API request and get the response
    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON data from the response
        data = response.json()

        # Extract the first result from the data
        result = data["features"][0]

        # Extract the latitude and longitude of the result
        latitude = result["geometry"]["coordinates"][1]
        longitude = result["geometry"]["coordinates"][0]

        print(f"Latitude: {latitude}, Longitude: {longitude}")
    else:
        print(f"Request failed with status code {response.status_code}")
    return f"{latitude}, {longitude}"



def OSMR (lat_1, long_1, lat_2, long_2):
    # call the OSMR API
    r = requests.get(f"http://router.project-osrm.org/route/v1/car/{long_1},{lat_1};{long_2},{lat_2}?overview=false""")
    # then you load the response using the json libray
    # by default you get only one alternative so you access 0-th element of the `routes`
    routes = json.loads(r.content)
    route_1 = routes.get("routes")[0]
    return pd.DataFrame(route_1)[["duration", "distance"]].values



def find_important_features (model):
    return pd.DataFrame(np.array([model.feature_names_in_, model.feature_importances_]).T, columns = ["Name", "Importance"]).sort_values(by = "Importance", ascending = False)



# def EDA_report (dist_data):
#     results = {}
#     #display(dist_data.describe())
#     results["describe"] = dist_data.describe()
#     num_cols = dist_data.shape[1]
#     num_x_cols = num_cols - 1
#     y_col = dist_data.columns[-1]
#     #print("Field correlation with SalePrice")
#     #print(dist_data.corr()[y_col].sort_values(ascending = False))
#     #print("-"*30)
#     #print()
#     results["corr"] = dist_data.corr()[y_col].sort_values(ascending = False)
#     X = dist_data.iloc[:, :num_x_cols]
#     y = dist_data[y_col]
#     #print("Random Forest Feature Importances")
#     RFR = RandomForestRegressor(random_state = 1)
#     RFR.fit(X, y)
#     results["important_features"] = find_important_features(RFR)
#     #print(find_important_features(RFR))
#     #print("-"*30)
#     #print()
#     model = smf.ols(formula = y_col + " ~ " + "+".join(dist_data.columns[:-1].tolist()), data = dist_data).fit()
#     summary = model.summary()
#     results["ols_summary"] = summary
#     #print(summary)
#     plt.figure(figsize = (14,14))
#     sns.heatmap(dist_data.corr())
#     return results

### NEW FUNCTION REPLACES ABOVE ###
def EDA_report (dist_data, show_heatmap = False):
    results = {}
    results["describe"] = dist_data.describe()
    num_cols = dist_data.shape[1]
    num_x_cols = num_cols - 1
    y_col = dist_data.columns[-1]
    results["corr"] = dist_data.corr()[y_col].sort_values(ascending = False)
    X = dist_data.iloc[:, :num_x_cols]
    y = dist_data[y_col]
    RFR = RandomForestRegressor(random_state = 3)
    RFR.fit(X, y)
    results["important_features"] = find_important_features(RFR)
    model = smf.ols(formula = y_col + " ~ " + "+".join(dist_data.columns[:-1].tolist()), data = dist_data).fit()
    summary = model.summary()
    results["ols_summary"] = summary
    if show_heatmap:
        plt.figure(figsize = (14,14))
        sns.heatmap(dist_data.corr())
    return results
    


#Distance function from one lat long to another
def get_dist(from_location, to_location):
    #print("from",from_location)
    from_loc = from_location * (np.pi)/180
    #print("to", to_location)
    to_loc = to_location * np.pi/180
    delta = from_loc - to_loc
    phi = (from_loc[0] + to_loc[0])/2
    #print(from_loc)
    #print(to_loc)
    return np.sqrt((np.cos(phi)*delta[1])**2+delta[0]**2)*3963.19



#Creating Automating function for analyzing distances with different Essential latlongs
def find_min_dist (df_house, df_biz):
    df_cross = df_house.join(df_biz, how = "cross", lsuffix = '_house', rsuffix = "_biz")
    df_cross["dist"] = df_cross.apply(lambda x: get_dist(np.array([x.Lat_house, x.Long_house]), np.array([x.Lat_biz, x.Long_biz])), axis = 1)
    df_subset = df_cross.loc[df_cross.groupby(["Service", "SaleID"]).dist.idxmin()]

    return (df_subset
            .pivot(columns = "Service", 
                    index = ["SaleID", "Prop_Addr", "SalePrice"], 
                    values = "dist")
            .join(df_subset.pivot(columns = "Service", index = "SaleID", values = "Name"),
                    lsuffix = "_dist", 
                    rsuffix = "_name")
            .reset_index()
            )

def save_state_pkl (k,v, desc):
    #TODO: Add code to create pickle file if it doesn't exist yet
    Ames_notebook_state = pickle.load(open("Ames_notebook_state.pkl", "rb"))
    Ames_notebook_state[k] = (v, desc)
    pickle.dump(Ames_notebook_state, open("Ames_notebook_state.pkl", "wb"))

def load_state_pkl ():
    return pickle.load(open("Ames_notebook_state.pkl", "rb"))

def remove_undeveloped_lots (df, df_house = None):
    df_copy = df.copy()
    if "GLA" not in df:
        df_copy = df_copy.join(df_house[["GLA", "YrSold_YYYY"]])
    return df_copy.query("YrBuilt <= YrSold_YYYY and YrBuilt != 0 and GLA > 0")
