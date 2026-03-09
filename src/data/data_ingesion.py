import pandas as pd
import numpy as np
from xgboost import data
from xgboost import data
from logger.logger import get_logger
import os

logger = get_logger(__name__)

def load_data(file_path:str)->pd.DataFrame:
    try:
        logger.info(f'loading data from {file_path}')
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f'file not found at {file_path}: {e}')
        raise
    except Exception as e:
        logger.error(f'error loading data from {file_path}: {e}')
        raise


def data_cleaning(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('starting data cleaning')
        data.columns = data.columns.str.lower()
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data
    except Exception as e:
        logger.error(f'an error accured while cleaning the data :{e}')
        raise


def removing_corr_and_irrelevent_col(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('removing highly correlated and irrelevent columns')
        # irrelevant column
        data.drop(columns=['customerID'],inplace=True)
        # due to the high correlation with monyhlycharges.
        data.drop(columns=['totalcharges'], inplace=True)
        return data
    except Exception as e:
        logger.error(f'an error accured while removing correlated and irrelevant columns :{e}')
        raise


def fixing_cols_data_types(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('fixing columns data types')
        cat_cols = [
        "gender","Partner","Dependents",
        "PhoneService","PaperlessBilling","Churn",
        "MultipleLines","TechSupport","StreamingTV",
        "OnlineBackup","DeviceProtection","StreamingMovies",
        "Contract","OnlineSecurity","InternetService",
        "PaymentMethod"
    ]
        # convert to category
        data[cat_cols] = data[cat_cols].astype("category")
        data["SeniorCitizen"] = data["SeniorCitizen"].map({0:"No",1:"Yes"}).astype("category")
        data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')

        return data
    except Exception as e:
        logger.error(f'an error accured while fixing columns data types :{e}')
        raise


def service_col_to_binary(data:pd.DataFrame)-> pd.DataFrame:
    try:
        logger.info('converting service columns to binary')
        # Convert service columns to binary (Yes=1, No and No internet service=0)
        service_cols = ['onlinesecurity', 'onlinebackup', 'deviceprotection','techsupport', 'streamingtv', 'streamingmovies']
        for col in service_cols:
            data[col] = (data[col] == 'Yes').astype(int)
        return data
    except Exception as e:
        logger.error(f'an error accured while converting service columns to binary :{e}')
        raise


def save_cleaned_data(data:pd.DataFrame,data_save_path:str)->None:
    try:
        logger.info(f"trying to save the cleaned data at: {data_save_path}")
        data_save_raw_path=os.path.join(data_save_path,'interim')
        os.makedirs(data_save_raw_path,exist_ok=True)
        data.to_csv(os.path.join(data_save_raw_path,'cleaned_data.csv'),index=False)
        logger.info(f'cleaned_data saved successfully at: {data_save_raw_path}')
    except Exception as e:
        logger.error(f'an error accured while saving the data :{e}')
        raise


def main():
    data=load_data('data/raw/Telecom_Churn.csv') 
    data=data_cleaning(data)
    col_rem_data=removing_corr_and_irrelevent_col(data)
    fixed_dtype_data=fixing_cols_data_types(col_rem_data)
    service_col_bin_data=service_col_to_binary(fixed_dtype_data)
    save_cleaned_data(service_col_bin_data,'data')


if __name__=="__main__":
    main()