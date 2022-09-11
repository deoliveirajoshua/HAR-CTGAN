#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import json
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[ ]:


all_data_file_name = "..\\aggregated_data\\aggregated_data.csv"
data = pd.read_csv(all_data_file_name)
data.drop(columns = ['UUID', "timestamp"], axis = 1, inplace = True)
for c in data.columns:
    if "label:" in c:
        print(c)


# In[ ]:


total_entries = len(data.index)
# remove features with this percentage of entries missing; otherwise, populate NaN with 0
print(f"Total Entries: {total_entries}")
print(f"Total Features: {len(data.columns)}")
row_thresh = round(.85 * len(data.columns))
data.dropna(axis = 0, inplace = True, thresh = row_thresh)
data.fillna(value = 0, inplace= True)
print(f"Total Entries: {len(data.index)}")
print(f"Total Features: {len(data.columns)}")

labels = ['label:LYING_DOWN', 'label:CLEANING', 'label:SLEEPING', 'label:SITTING', 'label:FIX_walking', 'label:FIX_running', 'label:BICYCLING']
discrete = []
numerical = []
for col in data.columns:
    if col in labels:
        continue
    elif "discrete:" in col or "label:" in col:
        discrete.append(col)
    else:
        numerical.append(col)


# In[ ]:


print(len(discrete))
print(discrete)
app_state = ['discrete:app_state:is_active', 'discrete:app_state:is_inactive', 'discrete:app_state:is_background', 'discrete:app_state:missing']
battery_state = ['discrete:battery_state:is_unknown', 'discrete:battery_state:is_unplugged', 'discrete:battery_state:is_not_charging', 'discrete:battery_state:is_discharging', 'discrete:battery_state:is_charging', 'discrete:battery_state:is_full', 'discrete:battery_state:missing']
battery_plugged = ['discrete:battery_plugged:is_ac', 'discrete:battery_plugged:is_usb', 'discrete:battery_plugged:is_wireless', 'discrete:battery_plugged:missing']
on_the_phone = ['discrete:on_the_phone:is_False', 'discrete:on_the_phone:is_True', 'discrete:on_the_phone:missing']
ringer_mode = ['discrete:ringer_mode:is_normal', 'discrete:ringer_mode:is_silent_no_vibrate', 'discrete:ringer_mode:is_silent_with_vibrate', 'discrete:ringer_mode:missing']
wifi_status = ['discrete:wifi_status:is_not_reachable', 'discrete:wifi_status:is_reachable_via_wifi', 'discrete:wifi_status:is_reachable_via_wwan', 'discrete:wifi_status:missing']
time_of_day = ['discrete:time_of_day:between0and6', 'discrete:time_of_day:between3and9', 'discrete:time_of_day:between6and12', 'discrete:time_of_day:between9and15', 'discrete:time_of_day:between12and18', 'discrete:time_of_day:between15and21', 'discrete:time_of_day:between18and24', 'discrete:time_of_day:between21and3']
PHONE = ['label:PHONE_IN_POCKET','label:PHONE_IN_HAND', 'label:PHONE_IN_BAG', 'label:PHONE_ON_TABLE']
non_binary_discrete_feats = [app_state, battery_state, battery_plugged, on_the_phone, ringer_mode, wifi_status, time_of_day, PHONE]
binary_discrete_feats = []
for feature in discrete:
    found = False
    for feat_list in non_binary_discrete_feats:
        if feature in feat_list:
            found = True
            break;
    if not found:
        binary_discrete_feats.append(feature)
        
print(non_binary_discrete_feats)
print(binary_discrete_feats)
print(len(non_binary_discrete_feats))
print(len(binary_discrete_feats))

feat_lengths = []
for feat in non_binary_discrete_feats:
    feat_lengths.append(len(feat))
print(feat_lengths)


# In[ ]:


print(len(numerical))
print(numerical)


# In[ ]:


data = data[data[labels].sum(axis = 1) == 1]
print(len(data.index))

non_binary_names = ['app_state', 'battery_state', 'battery_plugged', 'on_the_phone', 'ringer_mode', 'wifi_status', 'time_of_day', 'PHONE']
i = 0
for group in non_binary_discrete_feats:
    one_hotted = data[group]
    data[non_binary_names[i]]=one_hotted.values.argmax(1)
    i = i+1


# In[ ]:


importances = {}

agg_D = torch.tensor([0.]*len(discrete))
agg_N = torch.tensor([0.]*len(numerical))

discrete = binary_discrete_feats + non_binary_names
for idx, label in enumerate(labels):
        print(f"{idx+1} of {len(labels)}")
        X_d = data[discrete]
        X_n = data[numerical]
        Y = data[label]
        
        forest_d = RandomForestClassifier()
        forest_d.fit(X_d, Y)
        
        forest_n = RandomForestClassifier()
        forest_n.fit(X_n, Y)
        
        
        imp_d = forest_d.feature_importances_
        imp_n = forest_n.feature_importances_
        
        important_discrete = {}
        important_numerical = {}
        
        args_imp_d = np.argsort(imp_d)
        print(args_imp_d)
        
        args_imp_n = np.argsort(imp_n)
        print(args_imp_n)
        
        
        for idx, disc in enumerate(args_imp_d):
            important_discrete[discrete[int(disc)]] = int(idx+1)
            agg_D[disc] += idx
            
        for idx, numer in enumerate(args_imp_n):
            important_numerical[numerical[int(numer)]] = int(idx+1)
            agg_N[numer] += idx
        
        print(agg_D)
        print(agg_N)
        
        imp = {}
        imp['binary_discrete'] = important_discrete
        imp['numerical'] = important_numerical
        importances[label] = imp
        
        
        


# In[ ]:


idx_D = (-agg_D).argsort()
top_D = []
idx_N = (-agg_N).argsort()
top_N = []

for el in idx_D:
    top_D.append(discrete[el])
for el in idx_N:
    top_N.append(numerical[el])

print(top_D)
print(agg_D)
print("\n")
print(top_N)
print(agg_N)
print("\n")


# In[ ]:





# In[ ]:


Y = np.argmax(data[labels].to_numpy(), axis=1)
count = pd.Series(Y).value_counts()
print ("Element  Count")
print(count)


# In[ ]:


top_D = ['time_of_day', 'PHONE', 'label:OR_indoors', 'label:OR_exercise', 'ringer_mode', 'label:LOC_home', 'wifi_status', 'battery_state', 'label:OR_outside', 'label:DRIVE_-_I_M_THE_DRIVER', 'label:SURFING_THE_INTERNET', 'label:WITH_FRIENDS', 'label:TALKING', 'app_state', 'label:WATCHING_TV', 'label:AT_SCHOOL', 'label:COMPUTER_WORK', 'label:EATING', 'label:LOC_main_workplace', 'label:IN_A_CAR', 'label:OR_standing', 'battery_plugged', 'label:SHOPPING', 'label:COOKING', 'label:DRIVE_-_I_M_A_PASSENGER', 'label:WITH_CO-WORKERS', 'label:FIX_restaurant', 'label:IN_CLASS', 'label:ON_A_BUS', 'label:AT_THE_GYM', 'label:WASHING_DISHES', 'label:STROLLING', 'label:IN_A_MEETING', 'label:STAIRS_-_GOING_DOWN', 'label:DOING_LAUNDRY', 'label:STAIRS_-_GOING_UP', 'label:TOILET', 'label:BATHING_-_SHOWER', 'label:DRINKING__ALCOHOL_', 'on_the_phone', 'label:AT_A_BAR', 'label:DRESSING', 'label:GROOMING', 'label:SINGING', 'label:LAB_WORK', 'label:LOC_beach', 'label:ELEVATOR', 'label:AT_A_PARTY']
Y = np.argmax(data[labels].to_numpy(), axis=1)
count = pd.Series(Y).value_counts()
print ("Element  Count")
print(count)


# In[ ]:



weighted_f1_discrete = []
macro_f1_discrete = []
accuracy_discrete = []
weighted_f1_discrete_err = []
macro_f1_discrete_err = []
accuracy_discrete_err = []

for idx in range(len(top_D)):
    print(f"{idx+1}/{len(top_D)}")
    tmp_weighted = []
    tmp_macro = []
    tmp_accuracy = []
    X_d = data[top_D[:idx+1]].to_numpy()
    Y = np.argmax(data[labels].to_numpy(), axis=1)
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_d, Y, test_size=0.7)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPClassifier(max_iter=100)
        model.fit(X_d, Y)

        y_pred = model.predict(X_test)

        tmp_weighted.append(round(f1_score(y_test, y_pred, average='weighted'),4))
        tmp_macro.append(round(f1_score(y_test, y_pred, average='macro'),4))
        tmp_accuracy.append(round(accuracy_score(y_test, y_pred), 4))
    weighted_f1_discrete.append(round(np.mean(np.array(tmp_weighted)),3)) 
    macro_f1_discrete.append(round(np.mean(np.array(tmp_macro)),3))
    accuracy_discrete.append(round(np.mean(np.array(tmp_accuracy)),3))
    weighted_f1_discrete_err.append(round(np.std(np.array(tmp_weighted)),3))
    macro_f1_discrete_err.append(round(np.std(np.array(tmp_macro)),3))
    accuracy_discrete_err.append(round(np.std(np.array(tmp_accuracy)),3))
                   
print(weighted_f1_discrete)
print(macro_f1_discrete) 
print(accuracy_discrete)
print(weighted_f1_discrete_err)
print(macro_f1_discrete_err)
print(accuracy_discrete_err)


# In[ ]:


top_N = ['raw_magnet:3d:mean_y', 'raw_magnet:3d:mean_x', 'raw_acc:3d:mean_y', 'watch_acceleration:magnitude_stats:percentile75', 'raw_acc:magnitude_stats:percentile75', 'raw_magnet:3d:mean_z', 'audio_naive:mfcc1:mean', 'proc_gyro:3d:std_x', 'proc_gyro:magnitude_stats:percentile25', 'proc_gyro:magnitude_stats:percentile50', 'proc_gyro:magnitude_stats:percentile75', 'raw_acc:3d:mean_x', 'raw_magnet:magnitude_stats:percentile75', 'proc_gyro:magnitude_stats:mean', 'proc_gyro:3d:std_y', 'raw_magnet:magnitude_stats:mean', 'raw_magnet:magnitude_stats:percentile50', 'raw_acc:3d:std_x', 'watch_acceleration:magnitude_stats:percentile25', 'watch_acceleration:3d:mean_x', 'raw_acc:3d:mean_z', 'watch_acceleration:magnitude_stats:mean', 'raw_acc:magnitude_stats:percentile25', 'raw_acc:magnitude_stats:std', 'raw_magnet:magnitude_stats:percentile25', 'watch_acceleration:magnitude_stats:std', 'raw_acc:3d:std_z', 'watch_acceleration:magnitude_stats:percentile50', 'raw_acc:3d:std_y', 'proc_gyro:3d:std_z', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0', 'raw_acc:magnitude_stats:moment4', 'raw_acc:magnitude_spectrum:log_energy_band4', 'raw_acc:magnitude_spectrum:log_energy_band2', 'proc_gyro:magnitude_stats:std', 'lf_measurements:battery_level', 'raw_acc:magnitude_stats:mean', 'raw_acc:magnitude_stats:percentile50', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1', 'watch_acceleration:3d:mean_y', 'raw_acc:magnitude_spectrum:log_energy_band3', 'audio_naive:mfcc0:std', 'audio_naive:mfcc3:mean', 'watch_acceleration:magnitude_stats:time_entropy', 'location:min_altitude', 'watch_acceleration:magnitude_spectrum:log_energy_band2', 'audio_naive:mfcc2:mean', 'raw_magnet:magnitude_spectrum:log_energy_band4', 'lf_measurements:screen_brightness', 'audio_naive:mfcc7:mean', 'watch_acceleration:magnitude_stats:moment4', 'audio_properties:normalization_multiplier', 'raw_acc:magnitude_stats:time_entropy', 'location:max_altitude', 'audio_properties:max_abs_value', 'watch_acceleration:3d:mean_z', 'audio_naive:mfcc12:mean', 'audio_naive:mfcc5:mean', 'proc_gyro:magnitude_stats:moment4', 'watch_acceleration:magnitude_spectrum:log_energy_band3', 'audio_naive:mfcc4:mean', 'audio_naive:mfcc0:mean', 'audio_naive:mfcc6:mean', 'raw_acc:magnitude_spectrum:spectral_entropy', 'raw_magnet:avr_cosine_similarity_lag_range2', 'watch_acceleration:3d:std_x', 'proc_gyro:magnitude_stats:moment3', 'watch_acceleration:spectrum:x_log_energy_band2', 'audio_naive:mfcc8:mean', 'watch_acceleration:3d:std_z', 'watch_acceleration:magnitude_stats:value_entropy', 'watch_acceleration:magnitude_spectrum:spectral_entropy', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2', 'proc_gyro:magnitude_stats:time_entropy', 'raw_magnet:magnitude_stats:time_entropy', 'proc_gyro:magnitude_spectrum:log_energy_band4', 'location:diameter', 'audio_naive:mfcc9:mean', 'raw_acc:magnitude_spectrum:log_energy_band1', 'watch_acceleration:magnitude_spectrum:log_energy_band1', 'watch_acceleration:spectrum:x_log_energy_band4', 'watch_acceleration:magnitude_spectrum:log_energy_band4', 'location:log_diameter', 'raw_magnet:3d:std_z', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3', 'watch_acceleration:spectrum:y_log_energy_band4', 'raw_magnet:magnitude_spectrum:log_energy_band3', 'watch_acceleration:3d:std_y', 'raw_acc:magnitude_spectrum:log_energy_band0', 'watch_acceleration:magnitude_autocorrelation:normalized_ac', 'raw_magnet:3d:std_x', 'proc_gyro:magnitude_stats:value_entropy', 'raw_acc:magnitude_stats:moment3', 'watch_acceleration:spectrum:x_log_energy_band1', 'watch_acceleration:magnitude_stats:moment3', 'audio_naive:mfcc11:mean', 'proc_gyro:magnitude_spectrum:spectral_entropy', 'raw_magnet:avr_cosine_similarity_lag_range0', 'audio_naive:mfcc1:std', 'watch_acceleration:spectrum:y_log_energy_band3', 'audio_naive:mfcc2:std', 'raw_magnet:3d:std_y', 'raw_magnet:magnitude_stats:std', 'raw_magnet:3d:ro_xz', 'watch_acceleration:spectrum:x_log_energy_band3', 'watch_acceleration:spectrum:z_log_energy_band2', 'raw_magnet:magnitude_stats:value_entropy', 'location:num_valid_updates', 'audio_naive:mfcc10:mean', 'proc_gyro:3d:ro_xy', 'raw_magnet:avr_cosine_similarity_lag_range4', 'raw_acc:magnitude_stats:value_entropy', 'raw_magnet:avr_cosine_similarity_lag_range1', 'raw_acc:magnitude_autocorrelation:normalized_ac', 'watch_acceleration:spectrum:z_log_energy_band4', 'raw_acc:3d:ro_yz', 'watch_acceleration:spectrum:y_log_energy_band2', 'watch_acceleration:spectrum:x_log_energy_band0', 'raw_magnet:avr_cosine_similarity_lag_range3', 'raw_magnet:magnitude_stats:moment4', 'location:log_latitude_range', 'watch_heading:mean_sin', 'location:best_horizontal_accuracy', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4', 'proc_gyro:3d:mean_y', 'proc_gyro:magnitude_spectrum:log_energy_band0', 'watch_heading:mean_cos', 'audio_naive:mfcc3:std', 'raw_magnet:3d:ro_xy', 'watch_acceleration:spectrum:y_log_energy_band1', 'audio_naive:mfcc5:std', 'watch_acceleration:spectrum:z_log_energy_band3', 'location:max_speed', 'proc_gyro:3d:ro_xz', 'raw_magnet:magnitude_spectrum:log_energy_band2', 'location_quick_features:std_lat', 'watch_acceleration:spectrum:z_log_energy_band0', 'proc_gyro:3d:mean_x', 'watch_acceleration:spectrum:z_log_energy_band1', 'proc_gyro:3d:ro_yz', 'audio_naive:mfcc9:std', 'watch_heading:entropy_8bins', 'audio_naive:mfcc12:std', 'lf_measurements:pressure', 'raw_magnet:magnitude_stats:moment3', 'audio_naive:mfcc7:std', 'audio_naive:mfcc8:std', 'watch_heading:std_sin', 'watch_acceleration:magnitude_spectrum:log_energy_band0', 'audio_naive:mfcc11:std', 'proc_gyro:magnitude_spectrum:log_energy_band3', 'raw_magnet:3d:ro_yz', 'raw_acc:magnitude_autocorrelation:period', 'raw_magnet:magnitude_spectrum:log_energy_band1', 'proc_gyro:magnitude_spectrum:log_energy_band1', 'proc_gyro:magnitude_spectrum:log_energy_band2', 'audio_naive:mfcc10:std', 'location:min_speed', 'raw_acc:3d:ro_xz', 'audio_naive:mfcc6:std', 'watch_heading:mom4_sin', 'proc_gyro:3d:mean_z', 'audio_naive:mfcc4:std', 'location:log_longitude_range', 'watch_heading:std_cos', 'raw_magnet:magnitude_spectrum:spectral_entropy', 'raw_acc:3d:ro_xy', 'lf_measurements:light', 'watch_acceleration:magnitude_autocorrelation:period', 'location_quick_features:mean_abs_long_deriv', 'proc_gyro:magnitude_autocorrelation:normalized_ac', 'location_quick_features:mean_abs_lat_deriv', 'watch_heading:mom3_sin', 'raw_magnet:magnitude_autocorrelation:normalized_ac', 'label_source', 'watch_acceleration:3d:ro_xy', 'raw_magnet:magnitude_spectrum:log_energy_band0', 'location_quick_features:long_change', 'watch_acceleration:3d:ro_yz', 'watch_heading:mom3_cos', 'location_quick_features:std_long', 'location:best_vertical_accuracy', 'proc_gyro:magnitude_autocorrelation:period', 'location_quick_features:lat_change', 'watch_heading:mom4_cos', 'raw_magnet:magnitude_autocorrelation:period', 'watch_acceleration:3d:ro_xz', 'watch_acceleration:spectrum:y_log_energy_band0', 'lf_measurements:temperature_ambient', 'lf_measurements:relative_humidity', 'lf_measurements:proximity_cm', 'lf_measurements:proximity']


# In[ ]:


weighted_f1_continuous = []
macro_f1_continuous = []
accuracy_continuous = []
weighted_f1_continuous_err = []
macro_f1_continuous_err = []
accuracy_continuous_err = []

for idx in range(len(top_N)):
    print(f"{idx+1}/{len(top_N)}")
    tmp_weighted = []
    tmp_macro = []
    tmp_accuracy = []
    X_d = data[top_N[:idx+1]].to_numpy()
    Y = np.argmax(data[labels].to_numpy(), axis=1)
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_d, Y, test_size=0.7)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPClassifier(max_iter=100)
        model.fit(X_d, Y)

        y_pred = model.predict(X_test)

        tmp_weighted.append(round(f1_score(y_test, y_pred, average='weighted'),4))
        tmp_macro.append(round(f1_score(y_test, y_pred, average='macro'),4))
        tmp_accuracy.append(round(accuracy_score(y_test, y_pred), 4))
    weighted_f1_continuous.append(round(np.mean(np.array(tmp_weighted)),3)) 
    macro_f1_continuous.append(round(np.mean(np.array(tmp_macro)),3))
    accuracy_continuous.append(round(np.mean(np.array(tmp_accuracy)),3))
    weighted_f1_continuous_err.append(round(np.std(np.array(tmp_weighted)),3))
    macro_f1_continuous_err.append(round(np.std(np.array(tmp_macro)),3))
    accuracy_continuous_err.append(round(np.std(np.array(tmp_accuracy)),3))
                   
print(weighted_f1_continuous)
print(macro_f1_continuous) 
print(accuracy_continuous)
print(weighted_f1_continuous_err)
print(macro_f1_conintuous_err)
print(accuracy_continuous_err)


# In[ ]:


x_axis = np.arange(1, len(weighted_f1_discrete)+1, 1, dtype=int)

#plt.title("Selectwise Discrete Feature Selection")
plt.plot(x_axis, weighted_f1_discrete)
plt.fill_between(x_axis, np.array(weighted_f1_discrete) - np.array(weighted_f1_discrete_err), 
                 np.array(weighted_f1_discrete) + np.array(weighted_f1_discrete_err), color='blue', alpha=0.1, label='Weighted F1')
plt.legend()
plt.xlabel("Top-K Features")
plt.ylabel("HAR Performance")
plt.ylim(0., 1.)
plt.xlim(1, len(weighted_f1_discrete))
plt.grid()
plt.savefig('discrete_selectwise_weightedf1.pdf', dpi=300)  
plt.show()

#plt.title("Selectwise Discrete Feature Selection")
plt.plot(x_axis, macro_f1_discrete, color='orange')
plt.fill_between(x_axis, np.array(macro_f1_discrete) - np.array(macro_f1_discrete_err), 
                 np.array(macro_f1_discrete) + np.array(macro_f1_discrete_err), color='orange', alpha=0.3, label='Macro F1')
plt.legend()
plt.xlabel("Top-K Features")
plt.ylabel("HAR Performance")
plt.ylim(0., 1.)
plt.xlim(1, len(weighted_f1_discrete))
plt.grid()
plt.savefig('discrete_selectwise_macrof1.pdf', dpi=300)
plt.show()

#plt.title("Selectwise Discrete Feature Selection")
plt.plot(x_axis, accuracy_discrete, color='green')
plt.fill_between(x_axis, np.array(accuracy_discrete) - np.array(accuracy_discrete_err), 
                 np.array(accuracy_discrete) + np.array(accuracy_discrete_err), color='green', alpha=0.1, label='Accuracy')
plt.legend()
plt.xlabel("Top-K Features")
plt.ylabel("HAR Performance")
plt.ylim(0., 1.)
plt.xlim(1, len(weighted_f1_discrete))
plt.grid()
plt.savefig('discrete_selectwise_accuracy.pdf', dpi=300)  


# In[ ]:


x_axis = np.arange(1, len(weighted_f1_continuous)+1, 1, dtype=int)

#plt.title("Selectwise Continuous Feature Selection")
plt.plot(x_axis, weighted_f1_continuous)
plt.fill_between(x_axis, np.array(weighted_f1_continuous) - np.array(weighted_f1_continuous_err),np.array(weighted_f1_continuous) + np.array(weighted_f1_continuous_err), color='blue', alpha=0.1, label='Weighted F1')
plt.legend()
plt.xlabel("Top-K Features")
plt.ylabel("HAR Performance")
plt.ylim(0., 1.)
plt.xlim(1, len(weighted_f1_continuous))
plt.grid()
plt.savefig('continuous_selectwise_weightedf1.pdf', dpi=300)  
plt.show()

#plt.title("Selectwise Continuous Feature Selection")
plt.plot(x_axis, macro_f1_continuous, color='orange')
plt.fill_between(x_axis, np.array(macro_f1_continuous) - np.array(macro_f1_continuous_err), 
                 np.array(macro_f1_continuous) + np.array(macro_f1_continuous_err), color='orange', alpha=0.3, label='Macro F1')
plt.legend()
plt.xlabel("Top-K Features")
plt.ylabel("HAR Performance")
plt.ylim(0., 1.)
plt.xlim(1, len(weighted_f1_continuous))
plt.grid()
plt.savefig('continuous_selectwise_macrof1.pdf', dpi=300)  
plt.show()

#plt.title("Selectwise Continuous Feature Selection")
plt.plot(x_axis, accuracy_continuous, color='green')
plt.fill_between(x_axis, np.array(accuracy_continuous) - np.array(accuracy_continuous_err), 
                 np.array(accuracy_continuous) + np.array(accuracy_continuous_err), color='green', alpha=0.1, label='Accuracy')
plt.legend()
plt.xlabel("Top-K Features")
plt.ylabel("HAR Performance")
plt.ylim(0., 1.)
plt.xlim(1, len(weighted_f1_continuous))
plt.grid()
plt.savefig('continuous_selectwise_accuracy.pdf', dpi=300)  
plt.show()


# In[ ]:


data.to_csv("cleaned_data_featurized.csv")

