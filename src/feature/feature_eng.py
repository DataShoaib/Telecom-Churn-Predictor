import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from logger.logger import get_logger
import os

logger=get_logger(__name__)

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


def feature_creation(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('starting feature creation')    
        # from eda we get the month-to-month contract type is giving the strong signal for the churn 
        data['contract_risk']=data['contract'].apply(lambda x: 1 if x=='Month-to-month' else 0)
        # Customers with DSL or Fiber optic = 1, No internet = 0, because dls and fiber optic both are internet services
        data['has_internet'] = data['internetservice'].apply(lambda x: 1 if x != 'No' else 0)
        # from the `eda` we get that if the payment methos is elctronics strong signal of churn
        data['paymentmethod_risk']=data['paymentmethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
        # Is customer on auto payment? (most committed)->lower risk customer signal
        data['is_autopay'] = data['paymentmethod'].apply(lambda x: 1 if 'automatic' in x.lower() else 0)
        # from the `eda` we get that if the customer is seniorcitizen than strong signal of churn
        data['seniorcitizen_risk']=data['seniorcitizen'].apply(lambda x: 1 if x == 'Yes' else 0)
        # is the customer is new (less than 6-month = new = high churn risk)
        data['is_new_customer'] = (data['tenure'] <= 6).astype(int)
        # Loyal customer -> more than 2 years
        data['is_loyal_customer'] = (data['tenure'] >= 24).astype(int)
        # total services used by the customer (count of total services used by the customer)
        service_cols = ['onlinesecurity', 'onlinebackup', 'deviceprotection','techsupport', 'streamingtv', 'streamingmovies']
        data['total_services'] = data[service_cols].sum(axis=1)
        # sticky_customer-> hard to leave (using more than 3 services)
        data['is_sticky_user']  = (data['total_services'] >= 3).astype(int)

        logger.info('feature creation completed successfully')
        return data
    except Exception as e:
        logger.error(f'error during feature creation: {e}')
        raise


# Define column groups
binary_cols  = ['partner', 'dependents', 'phoneservice', 'paperlessbilling','seniorcitizen']
gender_col   = ['gender']
ohe_cols     = ['multiplelines', 'internetservice', 'contract', 'paymentmethod']
scaling_cols = ['tenure', 'monthlycharges']

# -----HELPER FUNCTIONS-----

# Yes → 1, No → 0
def encode_yes_no(X):
    return (X == 'Yes').astype(int)

# Male → 1, Female → 0
def encode_gender(X):
    return (X == 'Male').astype(int)

def column_preprocessor(data:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('starting feature processing')
        data['seniorcitizen_risk'] = data['seniorcitizen_risk'].astype(int)
        preprocessor = ColumnTransformer(
        transformers=[
            ('binary', FunctionTransformer(encode_yes_no),                     binary_cols),
            ('gender', FunctionTransformer(encode_gender),                     gender_col),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False,
                                     handle_unknown='ignore'),                 ohe_cols),
            ('scale',  StandardScaler(),                                       scaling_cols),
        ],
        remainder='passthrough'  
        )
        logger.info('feature processing completed successfully')
        return preprocessor
       
    except Exception as e:
        logger.error(f'error during feature processing: {e}')
        raise

def save_feature_created_data(data:pd.DataFrame,data_save_path:str)->None:
    try:
        logger.info(f"trying to save the feature-created data at: {data_save_path}")
        data_save_raw_path=os.path.join(data_save_path,'processed')
        os.makedirs(data_save_raw_path,exist_ok=True)
        data.to_csv(os.path.join(data_save_raw_path,'feature_eng_data.csv'),index=False)
        logger.info(f'feature-created data saved successfully at: {data_save_raw_path}')
    except Exception as e:
        logger.error(f'an error accured while saving the data :{e}')
        raise

def main():
    data=load_data('data/interim/cleaned_data.csv')
    feature_created_data=feature_creation(data)
    save_feature_created_data(feature_created_data,'data')


if __name__=='__main__':
    main()    