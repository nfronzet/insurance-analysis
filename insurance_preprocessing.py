# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:10:56 2019

@author: Nicola
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Imputer
import time

def preprocess(filepath_in, filepath_out):
    raw_data = pd.read_csv(filepath_in)
    
    start_time = time.time()
    print('Starting data preprocessing...')
    
    #removing redundant columns: 'RC' is always 1.0 and 'comune' is redundant due to the column 'provincia'
    print('Dropping redundant columns...')
    raw_data.drop(columns=['RC','comune', 'valore'],inplace=True)
    print('Removing incomplete tuples...')
    raw_data.dropna(subset = ['provincia','sesso','antifurto','alimentazione'], inplace=True)
    
    print('Imputing columns...')
    imputer = Imputer(strategy='mean')
    raw_data[['etaContraente','etaVeicolo','potenza','cilindrata']] = imputer.fit_transform(raw_data[['etaContraente','etaVeicolo','potenza','cilindrata']])
    #imputer = Imputer(strategy='mean', missing_values=0)
    #raw_data[['valore']] = imputer.fit_transform(raw_data[['valore']])
    
    #normalising numerical values
    print('Rescaling data...')
    minmax = MinMaxScaler()
    raw_data[['etaVeicolo','potenza','cilindrata']] = minmax.fit_transform(raw_data[['etaVeicolo','potenza','cilindrata']])
    std = MinMaxScaler()
    raw_data[['etaContraente']] = std.fit_transform(raw_data[['etaContraente']])
    
    #one-hot encoding of nominal categorical variables
    print('Encoding data...')
    ohe_data = pd.get_dummies(raw_data[['provincia','sesso','antifurto','alimentazione']])
    raw_data.drop(columns=['provincia','sesso','antifurto','alimentazione'],inplace=True)
    pp_data = ohe_data.join(raw_data)
    
    #categorical mapping of output variables
    print('Creating output maps...')
    general_map = {0.0:0, 1.0:1}
    pp_data['ASS'] = pp_data['ASS'].map(general_map)
    pp_data['TUT'] = pp_data['TUT'].map(general_map)
    pp_data['INC'] = pp_data['INC'].map(general_map)
    pp_data['FUR'] = pp_data['FUR'].map(general_map)
    pp_data['ESP'] = pp_data['ESP'].map(general_map)
    pp_data['CRI'] = pp_data['CRI'].map(general_map)
    pp_data['EVN'] = pp_data['EVN'].map(general_map)
    pp_data['INF'] = pp_data['INF'].map(general_map)
    
    print('Saving processed data...')
    export_csv = pp_data.to_csv (filepath_out, index = None, header=True)
    print('Data preprocessing complete. Operation completed in {:.2f} s'.format(time.time()-start_time))
 
