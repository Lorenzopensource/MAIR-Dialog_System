properties = {"area": "north",
              "food type":"indian",
              "price range": "cheap"}
import pandas as pd 
def lookup(properties,data_path,n_matches):
    df=pd.read("restaurant_info.csv")
    for key, value in properties.items():
        if key in df.columns:
            df = df[df[key].str.lower()==value]
    return df["restaurantname"].head(n_matches).tolist()

