from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np

import pandas as pd
import numpy as np
import matplotlib
print(matplotlib.__version__)
from sklearn import metrics
from csv import writer

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Descriptors
from rdkit import Chem
import rdkit
import random
import warnings
import time
from tabulate import tabulate


# from rdkit.Chem import Draw

# Variation of the RF model that is used to evaluate individual input molecules. Given parameters of RF model, it trains it and then can be used to obtain results for 
# molecules that want to have their behaviour checked by the model.

warnings.filterwarnings("ignore")

def data_separator(guide_set, actual_set, column_name, desired_value):
    new_array = np.zeros(shape=(1,8))
    new_array = pd.DataFrame(new_array)
    lena = 0

    for n, row in guide_set.iterrows():
        
        if (row [column_name] == desired_value):
            captured_row = actual_set[lena:(lena+1):]  # Grabbing the row was the problem -> SOLVED -> n was the original order of the row in spreadsheet. As such, it went to the non-existant location in a shorter list  
            new_array = np.vstack([new_array, captured_row])
        lena += 1

    new_array = new_array[1::]
    return new_array

# set_checker(train_set, test_set, data, beh_type) - given pandas DataFrames: train_set, test_set and data, along with the 
# integer, beh_type, the function checks if the percentage representation of a given behaviour type is the same in the
# train_set and data_set. If the proportions are maintained, the function passes and allows the rest of the program to run. 
# If proportions are not maintained, the program is stopped and error message is printed.
# set_checker: pandas Dataframe; pandas Dataframe; pandas Dataframe; Integer -> None
def set_checker(train_set, test_set, data, beh_type):

    if (round((test_set['Behaviour Type'].value_counts()[beh_type] + train_set['Behaviour Type'].value_counts()[beh_type]) / data.shape[0],3) == round((data['Behaviour Type'].value_counts()[beh_type])/ data.shape[0],3)):
        pass

    else:
        print("error detected when checking for behaviour type ", beh_type)
        exit()

# write_results_to_csv(MAE, std_MAE, feature_list, save_to, file_started) - given np.float64: MAE, std_MAE, along with 
# a list of strings, feature_list, a file address, save_to and a boolean, file_started, the function formats writes those values 
# to the .csv file, by appending line by line. 
# write_results_to_csv: np.float64; np.float64; listof str; file_address; Bool -> Write to .csv
        
def write_results_to_csv(MAE, std_MAE, feature_list, removed_descriptor, save_to, file_started, active_mode):
    file_header = ["Number of Features", "MAE", "std_MAE", "Added Descriptor", "Feature List"]
    
    single_row = [len(feature_list), MAE, std_MAE, removed_descriptor, feature_list]

    single_row_frame = pd.DataFrame(single_row, index=None)
    single_row_frame = single_row_frame.transpose()

    if file_started == True:
        single_row_frame.to_csv(save_to, mode=active_mode, index=False, header=None)

    elif file_started == False:
        single_row_frame.to_csv(save_to, mode=active_mode, index=False, header=file_header)

# performace_evaluator(performance_csv_path, MAE) - given an address of .csv file to which results are written: 
# performance_csv_path and np.float64, MAE, the function compares the performance of the current run of the model with
# the last run of the model, by directly comparing Mean Absolute Error obtained. Based on that it returns a boolean value
# signifying whether the model performed better or not.
# performace_evaluator: .csv file_path, np.float64 -> Bool

# def performance_evaluator(performance_csv_path, MAE):
#     performance_csv = pd.read_csv(performance_csv_path, encoding="latin1")
#     last_MAE = np.float64(performance_csv.tail(n=1)["MAE"])
#     print("last_MAE: ", last_MAE, " MAE: ", MAE)
#     if last_MAE > MAE:
#         return True
    
#     elif last_MAE <= MAE:
#         return False

    

# Leftover function from Cailum's Code - TODO: Ask Cailum what does it do.
def EC_fit(SV,a,b,c):
    return [a * x0**3 + b * x0**5 + c * x0**7 for x0 in SV]

# Builds and prints the results as the table in terminal.
def Build_results_table(data_set):
    header_names = ["MAE (V)", "N2", "MeOH", "A", "B", "C", "Positive", "Negative", "Overall"]
    print(tabulate(data_set, headers=header_names, tablefmt="grid"))

# Learning curve at the end of the study, right?
def performance_plot(data_file_path, save_to_path):
    performance_data = pd.read_csv(data_file_path, encoding="latin1")

    num_features_values = np.array(performance_data["Number of Features"])
    MAE_values = np.array(performance_data["MAE"])

    x = num_features_values
    y = MAE_values

    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.title("Performance as Function of Number of Features")
    plt.ylabel("MAE")
    plt.xlabel("Number of Features")
    plt.savefig(save_to_path)
    plt.clf()
    plt.close()

#Constants and values:

# Model Object 
rf = RandomForestRegressor(random_state=26, n_estimators=700, min_samples_split=2, min_samples_leaf=2, criterion='squared_error', bootstrap=True)
clf = rf

# Type of SMILES to be used. Either SMILES or ionicSMILES are options
key_SMILES = "SMILES"

# Input descriptors

# Path to precomputed data 
precomputed_data_address = r"/home/cmkstien/RF_models/ForDaniel/RF/Training Data Prep/Training Lab Data/Normalized_Working_Descriptors_-1.csv"


# Normalized data: /home/cmkstien/RF_models/ForDaniel/RF/Training Data Prep/Training Lab Data/Normalized_Working_Descriptors_-1.csv
# Prenormalized data: /home/cmkstien/RF_models/ForDaniel/RF/Training Data Prep/Training Lab Data/Working_Descriptors-mil.csv

seed_values = [20, 21, 22, 23, 24]

shuffle_inputs = True 

# Address to which to write the results of each run
save_to = r"/home/cmkstien/RF_models/ForDaniel/RF/RF/Emir's Molecules/Emir Dispersion Curves Predictions_all_descriptors.csv"

def evaluator_RF_model(model_object, key_SMILES, descriptors_to_use, precomputed_data_address, precomputed_target_data_address, save_to, seed_values, display_advanced_results):
    precomputed_data = pd.read_csv(precomputed_data_address, encoding="latin1")

    target_data = pd.read_csv(precomputed_target_data_address, encoding="latin1")
    
    # New Array to store the MAE at each seed
    Global_MAE_storage = []

    # ts - testing split proportion
    ts = 0.80

    SMILES_headers = ["SMILES", "ionicSMILES"]
    target_headers = ["1500", "2000", "2500", "3000", '3250', '3500', '3750', "4000"]
    essential_input_headers = ["Modifier", "Behaviour Type", "Chi2v", "Chi3v", "a_prop", "Chi0n", "MaxAbsPartialCharge", "Positive?", "NumRotatableBonds", \
                    "f_count", "MaxPartialCharge", "FractionCSP3", "NumHAcceptors", "MinPartialCharge", "NumHDonors", \
                    "MinAbsPartialCharge", "c_ar", "BalabanJ", "TPSA", "fr_amide", "Kappa3"]
    """
    # The Top 20 performing descriptors used as a starter in initial study

    essential_input_headers = ["Modifier", "Behaviour Type", "Chi2v", "Chi3v", "a_prop", "Chi0n", "MaxAbsPartialCharge", "Positive?", "NumRotatableBonds", \
                    "f_count", "MaxPartialCharge", "FractionCSP3", "NumHAcceptors", "MinPartialCharge", "NumHDonors", \
                    "MinAbsPartialCharge", "c_ar", "BalabanJ", "TPSA", "fr_amide", "Kappa3"]

                    ["Modifier", "Behaviour Type", "Chi2v", "Chi3v", "a_prop", "Chi0n"]
    """


    #["Modifier", "Behaviour Type", "Positive?"]
    added_descriptor_data = precomputed_data.copy()
    target_molecules_data = target_data.copy()
    

    added_descriptor_data = added_descriptor_data.drop(SMILES_headers, axis=1)
    added_descriptor_data = added_descriptor_data.drop(target_headers, axis=1)
    added_descriptor_data = added_descriptor_data.drop(essential_input_headers, axis=1)

    target_molecules_data = target_molecules_data.drop(SMILES_headers, axis=1)
    target_molecules_data = target_molecules_data.drop(target_headers, axis=1)
    target_molecules_data = target_molecules_data.drop(essential_input_headers, axis=1)

    # potential_descriptor_data_headers = potential_descriptor_data.columns.values.copy()

    core_headers_to_use = [key_SMILES]
    core_headers_to_use.extend(essential_input_headers)

# descriptors = np.zeros((potential_descriptor_data.shape[0], len(desired_features)))

    # Hard coded for now
    # xteT_preserve = np.zeros((1,117))
    # yteT_preserve = np.zeros((1,8))

    for i in seed_values:
        
        inputs = precomputed_data[core_headers_to_use]
        targets = precomputed_data[['1500', '2000','2500','3000','3250', '3500', '3750', '4000']]
        desired_molecules_inputs = target_data[core_headers_to_use]
        desired_molecules_targets = target_data[['1500', '2000','2500','3000','3250', '3500', '3750', '4000']]
        
# Add in to inputs the desired features to also include
    
    # Shortcut to trim down number of features just for sake of building architecture
        

        for descriptor_name in descriptors_to_use:
            
            if ((descriptor_name in essential_input_headers) or (descriptor_name in SMILES_headers) or (descriptor_name in target_headers)):
                pass

            else: 
                column_to_add = added_descriptor_data[[descriptor_name]]
                column_to_add_2 = target_molecules_data[[descriptor_name]]

                inputs = pd.concat([inputs, column_to_add], axis=1) 
                desired_molecules_inputs = pd.concat([desired_molecules_inputs, column_to_add_2], axis=1) 
        
        tn = targets.shape[1]
        inp = inputs.shape[1]

        ytrT = np.zeros((1,tn))
        ytrpT = np.zeros((1,tn))
        yteT = np.zeros((1,tn))
        ytepT = np.zeros((1,tn))
        xteT = np.zeros((1,inp))  

    
    #Modification of targets for splitting purposes
        split_target = targets.copy()
        split_target['Behaviour Type'] = precomputed_data['Behaviour Type']
        
    # split_target.sort_values(by= 'Behaviour Type')

    # Creation of split inputs and targets

        A_Type_inputs = inputs[inputs['Behaviour Type'] == 0]
        B_Type_inputs = inputs[inputs['Behaviour Type'] == 1]
        C_Type_inputs = inputs[inputs['Behaviour Type'] == 2]

        A_target = split_target[split_target['Behaviour Type'] == 0]
        B_target = split_target[split_target['Behaviour Type'] == 1]
        C_target = split_target[split_target['Behaviour Type'] == 2]

    # Train and Test Splitting
        
        A_x_train, A_x_test, A_y_train, A_y_test = train_test_split(A_Type_inputs, A_target, test_size = 1-ts, random_state = i)

        B_x_train, B_x_test, B_y_train, B_y_test = train_test_split(B_Type_inputs, B_target, test_size = 1-ts, random_state = i)

        C_x_train, C_x_test, C_y_train, C_y_test = train_test_split(C_Type_inputs, C_target, test_size = 1-ts, random_state = i)
        
    # Check if split is done well
        set_checker(A_x_train, A_x_test, A_Type_inputs, 0)
        set_checker(A_y_train, A_y_test, A_Type_inputs, 0)
        set_checker(B_x_train, B_x_test, B_Type_inputs, 1)
        set_checker(B_y_train, B_y_test, B_Type_inputs, 1)
        set_checker(C_x_train, C_x_test, C_Type_inputs, 2)
        set_checker(C_y_train, C_y_test, C_Type_inputs, 2)

        set_checker(A_x_train, A_x_test, precomputed_data, 0)
        set_checker(A_y_train, A_y_test, precomputed_data, 0)
        set_checker(B_x_train, B_x_test, precomputed_data, 1)
        set_checker(B_y_train, B_y_test, precomputed_data, 1)
        set_checker(C_x_train, C_x_test, precomputed_data, 2)
        set_checker(C_y_train, C_y_test, precomputed_data, 2)
        
    # recombine the sets 
        x_train = pd.concat([A_x_train, B_x_train, C_x_train])
        y_train = pd.concat([A_y_train, B_y_train, C_y_train])
        # x_test = pd.concat([A_x_test, B_x_test, C_x_test])
        # y_test = pd.concat([A_y_test, B_y_test, C_y_test])
       
        x_test = desired_molecules_inputs
        y_test = desired_molecules_targets   
   
    # Test the combined sets
        # set_checker (x_train, x_test, precomputed_data, 0)
        # set_checker (x_train, x_test, precomputed_data, 1)
        # set_checker (x_train, x_test, precomputed_data, 2)

        # set_checker (y_train, y_test, precomputed_data, 0)
        # set_checker (y_train, y_test, precomputed_data, 1)
        # set_checker (y_train, y_test, precomputed_data, 2)

    # Remove target parameter needed just for testing
        inputcols =  x_train.columns.to_list()

        # y_test = y_test.drop(labels=['Behaviour Type'], axis='columns') - Lacking the behaviour type as it is actual unknown
        y_train = y_train.drop(labels=['Behaviour Type'], axis='columns')
        x_train_r = x_train.drop(labels=[key_SMILES, 'Behaviour Type'], axis='columns')
        x_test_r = x_test.drop(labels=[key_SMILES, 'Behaviour Type'], axis='columns')

        input_columns = x_train_r.columns.to_list()
       
        model_object.fit(x_train_r, y_train)
        
        y_predict_rfr = model_object.predict((x_test_r))
        y_train_pred = model_object.predict((x_train_r))

        ## ytrT is y_train
        ## ytrpT is train_predictions
        ## yteT is y_test
        ## ytepT is test predictions
        ## xteT is x_test

        ## Stacks newest input on top of the older one
        ytrT = np.vstack([y_train, ytrT])
        ytrpT = np.vstack([y_train_pred, ytrpT])
        yteT = np.vstack([y_test, yteT])
        ytepT = np.vstack([y_predict_rfr, ytepT])
        xteT = np.vstack([x_test, xteT])

        # xteT_preserve = np.vstack([x_test, xteT_preserve])
        # yteT_preserve = np.vstack([y_test, yteT_preserve])
        
        MAE = metrics.mean_absolute_error(y_test,y_predict_rfr)
        r_square =np.corrcoef(y_test,y_predict_rfr)[0,1]**2
        print(np.round(MAE, 5), np.round(r_square, 8))
        Global_MAE_storage = np.append(Global_MAE_storage, MAE)

        def save_predictions_to_csv(prediction_values, target_data, csv_address):
            target_molecules_headers = ["SMILES", "Modifier", "Positive?"]
            prediction_headers = ["1500", "2000", "2500", "3000", "3250", "3500", "3750", "4000"]
            prediction_frame = pd.DataFrame(prediction_values, index=None, columns=prediction_headers)

            molecule_info_DataFrame = target_data[target_molecules_headers]

            final_frame = pd.concat([molecule_info_DataFrame,prediction_frame], axis=1)
            print(final_frame)

            final_frame.to_csv(save_to, index=None, mode="w")

 # Collects the resulting data and organizing it in a way that allows to display it as a table, allowing for easier viewing and analysis.
        def print_advanced_results():
            # Find Global stats
            table_data = [["1500V"], ["2000V"], ["2500V"], ["3000V"], ["3250V"], ["3500V"], ["3750V"], ["4000V"], ["Overall"]]

            MeOH_yteT = data_separator(xteT,yteT,'Modifier', 1)
            MeOH_ytepT = data_separator(xteT, ytepT, 'Modifier', 1)
            N2_yteT = data_separator(xteT,yteT, 'Modifier', 0)
            N2_ytepT = data_separator(xteT, ytepT, 'Modifier', 0)

            MAE_G_N2 = metrics.mean_absolute_error(N2_yteT,N2_ytepT)
            MAE_G_MeOH = metrics.mean_absolute_error(MeOH_yteT,MeOH_ytepT)

            Type_A_yteT = data_separator(xteT, yteT, 'Behaviour Type', 0)
            Type_A_ytepT = data_separator(xteT, ytepT, 'Behaviour Type', 0)
            Type_B_yteT = data_separator(xteT, yteT, 'Behaviour Type', 1)
            Type_B_ytepT = data_separator(xteT, ytepT, 'Behaviour Type', 1)
            Type_C_yteT = data_separator(xteT, yteT, 'Behaviour Type', 2)
            Type_C_ytepT = data_separator(xteT, ytepT, 'Behaviour Type', 2)

            MAE_G_A = metrics.mean_absolute_error(Type_A_yteT, Type_A_ytepT)
            MAE_G_B = metrics.mean_absolute_error(Type_B_yteT, Type_B_ytepT)
            MAE_G_C = metrics.mean_absolute_error(Type_C_yteT, Type_C_ytepT)

            pos_yteT = data_separator(xteT, yteT, 'Positive?', 0)
            pos_ytepT = data_separator(xteT, ytepT, 'Positive?', 0)
            neg_yteT = data_separator(xteT, yteT, 'Positive?', 1)
            neg_ytepT = data_separator(xteT, ytepT, 'Positive?', 1)

            MAE_G_pos = metrics.mean_absolute_error(pos_yteT, pos_ytepT)
            MAE_G_neg =  metrics.mean_absolute_error(neg_yteT, neg_ytepT)

            CVs = [1500, 2000, 2500, 3000, 3250, 3500, 3750, 4000]

            for i, CV in enumerate(CVs):
            
                MAE = metrics.mean_absolute_error(yteT[:,i],ytepT[:,i])
                MAE_N2 = metrics.mean_absolute_error(N2_yteT[:,i],N2_ytepT[:,i])
                MAE_MeOH = metrics.mean_absolute_error(MeOH_yteT[:,i],MeOH_ytepT[:,i])
                MAE_Type_A = metrics.mean_absolute_error(Type_A_yteT[:,i],Type_A_ytepT[:,i])
                MAE_Type_B = metrics.mean_absolute_error(Type_B_yteT[:,i],Type_B_ytepT[:,i])
                MAE_Type_C = metrics.mean_absolute_error(Type_C_yteT[:,i],Type_C_ytepT[:,i])
                MAE_pos = metrics.mean_absolute_error(pos_yteT[:,i],pos_ytepT[:,i])
                MAE_neg= metrics.mean_absolute_error(neg_yteT[:,i],neg_ytepT[:,i])

                table_single_row = [MAE_N2, MAE_MeOH, MAE_Type_A, MAE_Type_B, \
                                MAE_Type_C, MAE_pos , MAE_neg, MAE]
            

                table_data[i].extend(table_single_row)
            table_overall_row = [MAE_G_N2, MAE_G_MeOH, MAE_G_A, MAE_G_B, MAE_G_C, \
                            MAE_G_pos, MAE_G_neg, Final_MAE]
            table_data[-1].extend(table_overall_row)
            Build_results_table(table_data)

    ## Removing the zero column from the end
    ytrT = np.array(ytrT[:-1])
    ytrpT = np.array(ytrpT[:-1])
    yteT = np.array(yteT[:-1])
    ytepT = np.array(ytepT[:-1])
    xteT = np.array(xteT[:-1])
    # xteT_preserve = np.array(xteT_preserve[:-1])
    # yteT_preserve = np.array(yteT_preserve[:-1])
    # Change the columns here - use inputcols?
    xteT = pd.DataFrame(xteT, columns=inputcols)
#     xteT_preserve = pd.DataFrame(xteT_preserve, columns=inputcols) #['Chi0n','Chi2v','Chi3v','Modifier', 'a_prop', 'MaxAbsPartialCharge','Behaviour Type', 'Positive?'])

    MAE_t =metrics.mean_absolute_error(ytrT,ytrpT)
    print(MAE_t)
    MAE = metrics.mean_absolute_error(yteT,ytepT)
    r_square =np.corrcoef(yteT,ytepT)[0,1]**2

    Final_MAE = np.round(np.mean(Global_MAE_storage), 5)
    Final_std_MAE = np.round(np.std(Global_MAE_storage), 5)

    print("OVERALL ERROR")
    print(np.round(Final_MAE, 5), np.round(r_square, 8))
    print("********************************************")

    print("\n", "Mean of MAE for all seeds: ", Final_MAE) # Calculates the Mean of MAE from all seeds
    print("\n", "Standard Deviation of MAE for all seeds: ", Final_std_MAE) # Calculates the Standard Deviation of MAE from all seeds


    print("***************************************")
    # stack_test = np.transpose(np.vstack([ytrT,ytrpT]))
    # stack_train = np.transpose(np.vstack([yteT,ytepT]))

    save_predictions_to_csv(y_predict_rfr, target_data, save_to)


    if display_advanced_results == True:
        print_advanced_results()

    return Final_MAE, Final_std_MAE, input_columns #, xteT_preserve, yteT_preserve

rf = RandomForestRegressor(random_state=26, n_estimators=700, min_samples_split=2, min_samples_leaf=2, criterion='squared_error', bootstrap=True)
clf = rf
descriptors_to_use = ['Modifier', 'Chi2v', 'Chi3v', 'WPath', 'AATSC0pe', 'MID_C', 'NdssSe', 'BCUT2D_MRHI', 'ATSC1p', 'C3SP3', 'SpDiam_Dzp', 'n10FHRing', 'piPC2', 'TIC1', 'GATS7pe', 'SlogP_VSA2', 'SZ', 'ETA_dPsi_A', 'BCUTi-1l', 'fr_N_O', 'fr_COO', 'fr_benzene', 'Positive?', 'NssssC', 'SpMAD_D', 'SLogP', 'CIC1', 'fr_NH0', 'MPC10', 'BCUTdv-1l', 'AXp-5dv', 'Xch-3dv', 'MinAbsPartialCharge', 'ETA_eta_R', 'fr_urea', 'AATSC0c', 'fr_thiophene', 'ETA_dEpsilon_B', 'n8Ring', 'EState_VSA4', 'NumAromaticRings', 'AATSC0m', 'GATS2p', 'ETA_epsilon_4', 'NumAliphaticHeterocycles', 'AATS2i', 'SlogP_VSA6', 'ETA_epsilon_3', 'MPC3', 'SM1_Dzare', 'NsBr', 'SIC5', 'MolLogP', 'MATS4se', 'fr_Ar_NH', 'NaaCH', 'Kappa1', 'NaasC', 'SpMAD_Dzv', 'Zagreb2', 'BCUTpe-1h', 'JGI5', 'fr_C_S', 'NaaNH', 'C1SP2', 'NssNH2', 'AMID_h', 'GATS2s', 'SsNH3', 'ATSC3pe', 'fr_dihydropyridine', 'ATSC2c', 'nFAHRing', 'n7HRing', 'ZMIC0', 'fr_piperdine', 'amideB', 'AATS1i', 'fr_aryl_methyl', 'MID_O', 'NsNH3', 'BCUT2D_MRLOW', 'AATS7i', 'n7FRing', 'SsOH', 'BIC0', 'MID_N', 'fr_Imine', 'fr_Al_COO', 'HybRatio', 'ATS4m', 'fr_ester', 'AATS7pe', 'SMR_VSA3', 'GATS2m', 'BCUTs-1l', 'JGI4', 'nG12ARing', 'nAHRing', 'SRW06', 'AXp-7dv', 'nG12FARing', 'GATS1se', 'NssssN', 'NsSH', 'AMID_C', 'nBase', 'AATSC0se', 'BCUTm-1h', 'ATS2d', 'SRW10', 'NdNH']

all_descriptors = ['aa', 'a_prop', 't_order', 'c_ar', 'frac_C', 'TPSA_2', 'f_count', 'oh_count', 'cl_count', 'amideB', 'MaxAbsEStateIndex', \
    'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', \
    'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', \
    'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', \
    'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', \
    'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', \
    'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', \
    'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', \
    'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', \
    'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', \
    'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', \
    'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', \
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', \
    'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', \
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', \
    'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', \
    'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', \
    'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', \
    'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', \
    'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', \
    'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', \
    'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', \
    'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'ABC', 'ABCGG', 'nAcid', 'nBase', 'SpAbs_A', \
    'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A', 'VR3_A', 'nAromAtom', 'nAromBond', \
    'nAtom', 'nHeavyAtom', 'nSpiro', 'nBridgehead', 'nHetero', 'nH', 'nB', 'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl', 'nBr', 'nI', 'nX', \
    'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv', 'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS7dv', 'ATS8dv', 'ATS0d', 'ATS1d', 'ATS2d', 'ATS3d', 'ATS4d', \
    'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0s', 'ATS1s', 'ATS2s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s', 'ATS7s', 'ATS8s', 'ATS0Z', 'ATS1Z',\
    'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z', 'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', \
    'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v', 'ATS6v', 'ATS7v', 'ATS8v', 'ATS0se', 'ATS1se', 'ATS2se', 'ATS3se', \
    'ATS4se', 'ATS5se', 'ATS6se', 'ATS7se', 'ATS8se', 'ATS0pe', 'ATS1pe', 'ATS2pe', 'ATS3pe', 'ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', \
    'ATS8pe', 'ATS0are', 'ATS1are', 'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are', 'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p',\
    'ATS3p', 'ATS4p', 'ATS5p', 'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i', 'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', \
    'AATS0dv', 'AATS1dv', 'AATS2dv', 'AATS3dv', 'AATS4dv', 'AATS5dv', 'AATS6dv', 'AATS7dv', 'AATS8dv', 'AATS0d', 'AATS1d', 'AATS2d', 'AATS3d', \
    'AATS4d', 'AATS5d', 'AATS6d', 'AATS7d', 'AATS8d', 'AATS0s', 'AATS1s', 'AATS2s', 'AATS3s', 'AATS4s', 'AATS5s', 'AATS6s', 'AATS7s', 'AATS8s', \
    'AATS0Z', 'AATS1Z', 'AATS2Z', 'AATS3Z', 'AATS4Z', 'AATS5Z', 'AATS6Z', 'AATS7Z', 'AATS8Z', 'AATS0m', 'AATS1m', 'AATS2m', 'AATS3m', 'AATS4m', \
    'AATS5m', 'AATS6m', 'AATS7m', 'AATS8m', 'AATS0v', 'AATS1v', 'AATS2v', 'AATS3v', 'AATS4v', 'AATS5v', 'AATS6v', 'AATS7v', 'AATS8v', 'AATS0se', \
    'AATS1se', 'AATS2se', 'AATS3se', 'AATS4se', 'AATS5se', 'AATS6se', 'AATS7se', 'AATS8se', 'AATS0pe', 'AATS1pe', 'AATS2pe', 'AATS3pe', \
    'AATS4pe', 'AATS5pe', 'AATS6pe', 'AATS7pe', 'AATS8pe', 'AATS0are', 'AATS1are', 'AATS2are', 'AATS3are', 'AATS4are', 'AATS5are', 'AATS6are', \
    'AATS7are', 'AATS8are', 'AATS0p', 'AATS1p', 'AATS2p', 'AATS3p', 'AATS4p', 'AATS5p', 'AATS6p', 'AATS7p', 'AATS8p', 'AATS0i', 'AATS1i', 'AATS2i', \
    'AATS3i', 'AATS4i', 'AATS5i', 'AATS6i', 'AATS7i', 'AATS8i', 'ATSC0c', 'ATSC1c', 'ATSC2c', 'ATSC3c', 'ATSC4c', 'ATSC5c', 'ATSC6c', 'ATSC7c', \
    'ATSC8c', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC0d', 'ATSC1d', 'ATSC2d', \
    'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0s', 'ATSC1s', 'ATSC2s', 'ATSC3s', 'ATSC4s', 'ATSC5s', 'ATSC6s', 'ATSC7s', \
    'ATSC8s', 'ATSC0Z', 'ATSC1Z', 'ATSC2Z', 'ATSC3Z', 'ATSC4Z', 'ATSC5Z', 'ATSC6Z', 'ATSC7Z', 'ATSC8Z', 'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', \
    'ATSC4m', 'ATSC5m', 'ATSC6m', 'ATSC7m', 'ATSC8m', 'ATSC0v', 'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', \
    'ATSC0se', 'ATSC1se', 'ATSC2se', 'ATSC3se', 'ATSC4se', 'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se', 'ATSC0pe', 'ATSC1pe', 'ATSC2pe', \
    'ATSC3pe', 'ATSC4pe', 'ATSC5pe', 'ATSC6pe', 'ATSC7pe', 'ATSC8pe', 'ATSC0are', 'ATSC1are', 'ATSC2are', 'ATSC3are', 'ATSC4are', 'ATSC5are', \
    'ATSC6are', 'ATSC7are', 'ATSC8are', 'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p', 'ATSC7p', 'ATSC8p', 'ATSC0i', \
    'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i', 'AATSC0c', 'AATSC1c', 'AATSC2c', 'AATSC3c', 'AATSC4c', \
    'AATSC5c', 'AATSC6c', 'AATSC7c', 'AATSC8c', 'AATSC0dv', 'AATSC1dv', 'AATSC2dv', 'AATSC3dv', 'AATSC4dv', 'AATSC5dv', 'AATSC6dv', 'AATSC7dv', \
    'AATSC8dv', 'AATSC0d', 'AATSC1d', 'AATSC2d', 'AATSC3d', 'AATSC4d', 'AATSC5d', 'AATSC6d', 'AATSC7d', 'AATSC8d', 'AATSC0s', 'AATSC1s', \
    'AATSC2s', 'AATSC3s', 'AATSC4s', 'AATSC5s', 'AATSC6s', 'AATSC7s', 'AATSC8s', 'AATSC0Z', 'AATSC1Z', 'AATSC2Z', 'AATSC3Z', 'AATSC4Z', \
    'AATSC5Z', 'AATSC6Z', 'AATSC7Z', 'AATSC8Z', 'AATSC0m', 'AATSC1m', 'AATSC2m', 'AATSC3m', 'AATSC4m', 'AATSC5m', 'AATSC6m', 'AATSC7m', \
    'AATSC8m', 'AATSC0v', 'AATSC1v', 'AATSC2v', 'AATSC3v', 'AATSC4v', 'AATSC5v', 'AATSC6v', 'AATSC7v', 'AATSC8v', 'AATSC0se', 'AATSC1se', \
    'AATSC2se', 'AATSC3se', 'AATSC4se', 'AATSC5se', 'AATSC6se', 'AATSC7se', 'AATSC8se', 'AATSC0pe', 'AATSC1pe', 'AATSC2pe', 'AATSC3pe', \
    'AATSC4pe', 'AATSC5pe', 'AATSC6pe', 'AATSC7pe', 'AATSC8pe', 'AATSC0are', 'AATSC1are', 'AATSC2are', 'AATSC3are', 'AATSC4are', 'AATSC5are', \
    'AATSC6are', 'AATSC7are', 'AATSC8are', 'AATSC0p', 'AATSC1p', 'AATSC2p', 'AATSC3p', 'AATSC4p', 'AATSC5p', 'AATSC6p', 'AATSC7p', 'AATSC8p', \
    'AATSC0i', 'AATSC1i', 'AATSC2i', 'AATSC3i', 'AATSC4i', 'AATSC5i', 'AATSC6i', 'AATSC7i', 'AATSC8i', 'MATS1c', 'MATS2c', 'MATS3c', 'MATS4c', \
    'MATS5c', 'MATS6c', 'MATS7c', 'MATS8c', 'MATS1dv', 'MATS2dv', 'MATS3dv', 'MATS4dv', 'MATS5dv', 'MATS6dv', 'MATS7dv', 'MATS8dv', 'MATS1d', \
    'MATS2d', 'MATS3d', 'MATS4d', 'MATS5d', 'MATS6d', 'MATS7d', 'MATS8d', 'MATS1s', 'MATS2s', 'MATS3s', 'MATS4s', 'MATS5s', 'MATS6s', 'MATS7s', \
    'MATS8s', 'MATS1Z', 'MATS2Z', 'MATS3Z', 'MATS4Z', 'MATS5Z', 'MATS6Z', 'MATS7Z', 'MATS8Z', 'MATS1m', 'MATS2m', 'MATS3m', 'MATS4m', 'MATS5m', \
    'MATS6m', 'MATS7m', 'MATS8m', 'MATS1v', 'MATS2v', 'MATS3v', 'MATS4v', 'MATS5v', 'MATS6v', 'MATS7v', 'MATS8v', 'MATS1se', 'MATS2se', 'MATS3se', \
    'MATS4se', 'MATS5se', 'MATS6se', 'MATS7se', 'MATS8se', 'MATS1pe', 'MATS2pe', 'MATS3pe', 'MATS4pe', 'MATS5pe', 'MATS6pe', 'MATS7pe', 'MATS8pe', \
    'MATS1are', 'MATS2are', 'MATS3are', 'MATS4are', 'MATS5are', 'MATS6are', 'MATS7are', 'MATS8are', 'MATS1p', 'MATS2p', 'MATS3p', 'MATS4p', 'MATS5p', \
    'MATS6p', 'MATS7p', 'MATS8p', 'MATS1i', 'MATS2i', 'MATS3i', 'MATS4i', 'MATS5i', 'MATS6i', 'MATS7i', 'MATS8i', 'GATS1c', 'GATS2c', 'GATS3c', \
    'GATS4c', 'GATS5c', 'GATS6c', 'GATS7c', 'GATS8c', 'GATS1dv', 'GATS2dv', 'GATS3dv', 'GATS4dv', 'GATS5dv', 'GATS6dv', 'GATS7dv', 'GATS8dv', \
    'GATS1d', 'GATS2d', 'GATS3d', 'GATS4d', 'GATS5d', 'GATS6d', 'GATS7d', 'GATS8d', 'GATS1s', 'GATS2s', 'GATS3s', 'GATS4s', 'GATS5s', 'GATS6s', \
    'GATS7s', 'GATS8s', 'GATS1Z', 'GATS2Z', 'GATS3Z', 'GATS4Z', 'GATS5Z', 'GATS6Z', 'GATS7Z', 'GATS8Z', 'GATS1m', 'GATS2m', 'GATS3m', 'GATS4m', \
    'GATS5m', 'GATS6m', 'GATS7m', 'GATS8m', 'GATS1v', 'GATS2v', 'GATS3v', 'GATS4v', 'GATS5v', 'GATS6v', 'GATS7v', 'GATS8v', 'GATS1se', 'GATS2se', \
    'GATS3se', 'GATS4se', 'GATS5se', 'GATS6se', 'GATS7se', 'GATS8se', 'GATS1pe', 'GATS2pe', 'GATS3pe', 'GATS4pe', 'GATS5pe', 'GATS6pe', 'GATS7pe', \
    'GATS8pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS4are', 'GATS5are', 'GATS6are', 'GATS7are', 'GATS8are', 'GATS1p', 'GATS2p', 'GATS3p', 'GATS4p', \
    'GATS5p', 'GATS6p', 'GATS7p', 'GATS8p', 'GATS1i', 'GATS2i', 'GATS3i', 'GATS4i', 'GATS5i', 'GATS6i', 'GATS7i', 'GATS8i', 'BCUTc-1h', 'BCUTc-1l', \
    'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTs-1h', 'BCUTs-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', \
    'BCUTse-1h', 'BCUTse-1l', 'BCUTpe-1h', 'BCUTpe-1l', 'BCUTare-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l', 'BCUTi-1h', 'BCUTi-1l', 'SpAbs_DzZ', \
    'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', \
    'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', \
    'VR3_Dzm', 'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv', 'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', \
    'VR2_Dzv', 'VR3_Dzv', 'SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', \
    'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe', 'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe', \
    'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', \
    'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', \
    'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', \
    'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'nBonds', \
    'nBondsO', 'nBondsS', 'nBondsD', 'nBondsT', 'nBondsA', 'nBondsM', 'nBondsKS', 'nBondsKD', 'RNCG', 'RPCG', 'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', \
    'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'C4SP3', 'HybRatio', 'FCSP3', 'Xch-3d', 'Xch-4d', 'Xch-5d', 'Xch-6d', 'Xch-7d', 'Xch-3dv', 'Xch-4dv', \
    'Xch-5dv', 'Xch-6dv', 'Xch-7dv', 'Xc-3d', 'Xc-4d', 'Xc-5d', 'Xc-6d', 'Xc-3dv', 'Xc-4dv', 'Xc-5dv', 'Xc-6dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', \
    'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d', 'Xp-1d', 'Xp-2d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d', 'AXp-1d', 'AXp-2d', 'AXp-3d', \
    'AXp-4d', 'AXp-5d', 'AXp-6d', 'AXp-7d', 'Xp-0dv', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-0dv', 'AXp-1dv', \
    'AXp-2dv', 'AXp-3dv', 'AXp-4dv', 'AXp-5dv', 'AXp-6dv', 'AXp-7dv', 'SZ', 'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', \
    'Mpe', 'Mare', 'Mp', 'Mi', 'SpAbs_D', 'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D', \
    'NsLi', 'NssBe', 'NssssBe', 'NssBH', 'NsssB', 'NssssB', 'NsCH3', 'NdCH2', 'NssCH2', 'NtCH', 'NdsCH', 'NaaCH', 'NsssCH', 'NddC', 'NtsC', 'NdssC', \
    'NaasC', 'NaaaC', 'NssssC', 'NsNH3', 'NsNH2', 'NssNH2', 'NdNH', 'NssNH', 'NaaNH', 'NtN', 'NsssNH', 'NdsN', 'NaaN', 'NsssN', 'NddsN', 'NaasN', \
    'NssssN', 'NsOH', 'NdO', 'NssO', 'NaaO', 'NsF', 'NsSiH3', 'NssSiH2', 'NsssSiH', 'NssssSi', 'NsPH2', 'NssPH', 'NsssP', 'NdsssP', 'NsssssP', 'NsSH', \
    'NdS', 'NssS', 'NaaS', 'NdssS', 'NddssS', 'NsCl', 'NsGeH3', 'NssGeH2', 'NsssGeH', 'NssssGe', 'NsAsH2', 'NssAsH', 'NsssAs', 'NsssdAs', 'NsssssAs', \
    'NsSeH', 'NdSe', 'NssSe', 'NaaSe', 'NdssSe', 'NddssSe', 'NsBr', 'NsSnH3', 'NssSnH2', 'NsssSnH', 'NssssSn', 'NsI', 'NsPbH3', 'NssPbH2', 'NsssPbH', \
    'NssssPb', 'SsLi', 'SssBe', 'SssssBe', 'SssBH', 'SsssB', 'SssssB', 'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SddC', 'StsC', \
    'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsNH3', 'SsNH2', 'SssNH2', 'SdNH', 'SssNH', 'SaaNH', 'StN', 'SsssNH', 'SdsN', 'SaaN', 'SsssN', 'SddsN', 'SaasN', \
    'SssssN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SsSiH3', 'SssSiH2', 'SsssSiH', 'SssssSi', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SsssssP', 'SsSH', 'SdS', \
    'SssS', 'SaaS', 'SdssS', 'SddssS', 'SsCl', 'SsGeH3', 'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs', 'SsssdAs', 'SsssssAs', 'SsSeH', 'SdSe', \
    'SssSe', 'SaaSe', 'SdssSe', 'SddssSe', 'SsBr', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsI', 'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'ECIndex', \
    'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns', \
    'ETA_beta_ns_d', 'AETA_beta_ns_d', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_eta_R', 'AETA_eta_R', 'ETA_eta_RL', 'AETA_eta_RL', 'ETA_eta_F', \
    'AETA_eta_F', 'ETA_eta_FL', 'AETA_eta_FL', 'ETA_eta_B', 'AETA_eta_B', 'ETA_eta_BR', 'AETA_eta_BR', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_1', \
    'ETA_epsilon_2', 'ETA_epsilon_3', 'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_A', 'ETA_dEpsilon_B', 'ETA_dEpsilon_C', 'ETA_dEpsilon_D', 'ETA_dBeta', \
    'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'ETA_dPsi_B', 'fragCpx', 'fMF', 'nHBAcc', 'nHBDon', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', \
    'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', \
    'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'Kier1', 'Kier2', 'Kier3', \
    'Lipinski', 'GhoseFilter', 'FilterItLogS', 'VMcGowan', 'MDEC-22', 'MDEC-23', 'MDEC-33', 'MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C', 'AMID_C', 'MID_N', \
    'AMID_N', 'MID_O', 'AMID_O', 'MID_X', 'AMID_X', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', \
    'piPC4', 'piPC5', 'piPC6', 'piPC7', 'piPC8', 'piPC9', 'piPC10', 'TpiPC10', 'apol', 'bpol', 'nRing', 'n3Ring', 'n4Ring', 'n5Ring', 'n6Ring', 'n7Ring', \
    'n8Ring', 'n9Ring', 'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nHRing', 'n3HRing', 'n4HRing', 'n5HRing', 'n6HRing', 'n7HRing', 'n8HRing', 'n9HRing', \
    'n10HRing', 'n11HRing', 'n12HRing', 'nG12HRing', 'naRing', 'n3aRing', 'n4aRing', 'n5aRing', 'n6aRing', 'n7aRing', 'n8aRing', 'n9aRing', 'n10aRing', 'n11aRing', \
    'n12aRing', 'nG12aRing', 'naHRing', 'n3aHRing', 'n4aHRing', 'n5aHRing', 'n6aHRing', 'n7aHRing', 'n8aHRing', 'n9aHRing', 'n10aHRing', 'n11aHRing', 'n12aHRing', \
    'nG12aHRing', 'nARing', 'n3ARing', 'n4ARing', 'n5ARing', 'n6ARing', 'n7ARing', 'n8ARing', 'n9ARing', 'n10ARing', 'n11ARing', 'n12ARing', 'nG12ARing', 'nAHRing', \
    'n3AHRing', 'n4AHRing', 'n5AHRing', 'n6AHRing', 'n7AHRing', 'n8AHRing', 'n9AHRing', 'n10AHRing', 'n11AHRing', 'n12AHRing', 'nG12AHRing', 'nFRing', 'n4FRing', \
    'n5FRing', 'n6FRing', 'n7FRing', 'n8FRing', 'n9FRing', 'n10FRing', 'n11FRing', 'n12FRing', 'nG12FRing', 'nFHRing', 'n4FHRing', 'n5FHRing', 'n6FHRing', 'n7FHRing', \
    'n8FHRing', 'n9FHRing', 'n10FHRing', 'n11FHRing', 'n12FHRing', 'nG12FHRing', 'nFaRing', 'n4FaRing', 'n5FaRing', 'n6FaRing', 'n7FaRing', 'n8FaRing', 'n9FaRing', \
    'n10FaRing', 'n11FaRing', 'n12FaRing', 'nG12FaRing', 'nFaHRing', 'n4FaHRing', 'n5FaHRing', 'n6FaHRing', 'n7FaHRing', 'n8FaHRing', 'n9FaHRing', 'n10FaHRing', \
    'n11FaHRing', 'n12FaHRing', 'nG12FaHRing', 'nFARing', 'n4FARing', 'n5FARing', 'n6FARing', 'n7FARing', 'n8FARing', 'n9FARing', 'n10FARing', 'n11FARing', 'n12FARing', \
    'nG12FARing', 'nFAHRing', 'n4FAHRing', 'n5FAHRing', 'n6FAHRing', 'n7FAHRing', 'n8FAHRing', 'n9FAHRing', 'n10FAHRing', 'n11FAHRing', 'n12FAHRing', 'nG12FAHRing', \
    'nRot', 'RotRatio', 'SLogP', 'SMR', 'TopoPSA(NO)', 'TopoPSA', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'GGI6', 'GGI7', 'GGI8', 'GGI9', 'GGI10', 'JGI1', 'JGI2', \
    'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9', 'JGI10', 'JGT10', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'Vabc', 'VAdjMat', 'MWC01', \
    'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW03', 'SRW04', 'SRW05', 'SRW06', 'SRW07', 'SRW08', 'SRW09', \
    'SRW10', 'TSRW10', 'MW', 'AMW', 'WPath', 'WPol', 'Zagreb1', 'Zagreb2', 'mZagreb1', 'mZagreb2']

descriptors_to_use = all_descriptors

data_address = r"/home/cmkstien/RF_models/ForDaniel/RF/Training Data Prep/Training Lab Data/Working_Descriptors-mil.csv"

results = evaluator_RF_model(clf, "SMILES", descriptors_to_use, data_address, target_molecules_address, save_to, [20,21,22,23,24], False)