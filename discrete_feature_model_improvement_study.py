#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("..\\feature_importance\\cleaned_data_featurized.csv")

labels = ['label:LYING_DOWN', 'label:CLEANING', 'label:SLEEPING', 'label:SITTING', 'label:FIX_walking', 'label:FIX_running', 'label:BICYCLING']
discrete = ['time_of_day', 'PHONE', 'label:OR_indoors', 'label:OR_exercise', 'ringer_mode', 'label:LOC_home', 'wifi_status', 'battery_state', 'label:OR_outside', 'label:DRIVE_-_I_M_THE_DRIVER', 'label:SURFING_THE_INTERNET', 'label:WITH_FRIENDS', 'label:TALKING', 'app_state', 'label:WATCHING_TV', 'label:AT_SCHOOL', 'label:COMPUTER_WORK', 'label:EATING', 'label:LOC_main_workplace', 'label:IN_A_CAR', 'label:OR_standing', 'battery_plugged', 'label:SHOPPING', 'label:COOKING', 'label:DRIVE_-_I_M_A_PASSENGER', 'label:WITH_CO-WORKERS', 'label:FIX_restaurant', 'label:IN_CLASS', 'label:ON_A_BUS', 'label:AT_THE_GYM', 'label:WASHING_DISHES', 'label:STROLLING', 'label:IN_A_MEETING', 'label:STAIRS_-_GOING_DOWN', 'label:DOING_LAUNDRY', 'label:STAIRS_-_GOING_UP', 'label:TOILET', 'label:BATHING_-_SHOWER', 'label:DRINKING__ALCOHOL_', 'on_the_phone', 'label:AT_A_BAR', 'label:DRESSING', 'label:GROOMING', 'label:SINGING', 'label:LAB_WORK', 'label:LOC_beach', 'label:ELEVATOR', 'label:AT_A_PARTY']
numerical = ['raw_magnet:3d:mean_y', 'raw_magnet:3d:mean_x', 'raw_acc:3d:mean_y', 'watch_acceleration:magnitude_stats:percentile75', 'raw_acc:magnitude_stats:percentile75', 'raw_magnet:3d:mean_z', 'audio_naive:mfcc1:mean', 'proc_gyro:3d:std_x', 'proc_gyro:magnitude_stats:percentile25', 'proc_gyro:magnitude_stats:percentile50', 'proc_gyro:magnitude_stats:percentile75', 'raw_acc:3d:mean_x', 'raw_magnet:magnitude_stats:percentile75', 'proc_gyro:magnitude_stats:mean', 'proc_gyro:3d:std_y', 'raw_magnet:magnitude_stats:mean', 'raw_magnet:magnitude_stats:percentile50', 'raw_acc:3d:std_x', 'watch_acceleration:magnitude_stats:percentile25', 'watch_acceleration:3d:mean_x', 'raw_acc:3d:mean_z', 'watch_acceleration:magnitude_stats:mean', 'raw_acc:magnitude_stats:percentile25', 'raw_acc:magnitude_stats:std', 'raw_magnet:magnitude_stats:percentile25', 'watch_acceleration:magnitude_stats:std', 'raw_acc:3d:std_z', 'watch_acceleration:magnitude_stats:percentile50', 'raw_acc:3d:std_y', 'proc_gyro:3d:std_z', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0', 'raw_acc:magnitude_stats:moment4', 'raw_acc:magnitude_spectrum:log_energy_band4', 'raw_acc:magnitude_spectrum:log_energy_band2', 'proc_gyro:magnitude_stats:std', 'lf_measurements:battery_level', 'raw_acc:magnitude_stats:mean', 'raw_acc:magnitude_stats:percentile50', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1', 'watch_acceleration:3d:mean_y', 'raw_acc:magnitude_spectrum:log_energy_band3', 'audio_naive:mfcc0:std', 'audio_naive:mfcc3:mean', 'watch_acceleration:magnitude_stats:time_entropy', 'location:min_altitude', 'watch_acceleration:magnitude_spectrum:log_energy_band2', 'audio_naive:mfcc2:mean', 'raw_magnet:magnitude_spectrum:log_energy_band4', 'lf_measurements:screen_brightness', 'audio_naive:mfcc7:mean', 'watch_acceleration:magnitude_stats:moment4', 'audio_properties:normalization_multiplier', 'raw_acc:magnitude_stats:time_entropy', 'location:max_altitude', 'audio_properties:max_abs_value', 'watch_acceleration:3d:mean_z', 'audio_naive:mfcc12:mean', 'audio_naive:mfcc5:mean', 'proc_gyro:magnitude_stats:moment4', 'watch_acceleration:magnitude_spectrum:log_energy_band3', 'audio_naive:mfcc4:mean', 'audio_naive:mfcc0:mean', 'audio_naive:mfcc6:mean', 'raw_acc:magnitude_spectrum:spectral_entropy', 'raw_magnet:avr_cosine_similarity_lag_range2', 'watch_acceleration:3d:std_x', 'proc_gyro:magnitude_stats:moment3', 'watch_acceleration:spectrum:x_log_energy_band2', 'audio_naive:mfcc8:mean', 'watch_acceleration:3d:std_z', 'watch_acceleration:magnitude_stats:value_entropy', 'watch_acceleration:magnitude_spectrum:spectral_entropy', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2', 'proc_gyro:magnitude_stats:time_entropy', 'raw_magnet:magnitude_stats:time_entropy', 'proc_gyro:magnitude_spectrum:log_energy_band4', 'location:diameter', 'audio_naive:mfcc9:mean', 'raw_acc:magnitude_spectrum:log_energy_band1', 'watch_acceleration:magnitude_spectrum:log_energy_band1', 'watch_acceleration:spectrum:x_log_energy_band4', 'watch_acceleration:magnitude_spectrum:log_energy_band4', 'location:log_diameter', 'raw_magnet:3d:std_z', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3', 'watch_acceleration:spectrum:y_log_energy_band4', 'raw_magnet:magnitude_spectrum:log_energy_band3', 'watch_acceleration:3d:std_y', 'raw_acc:magnitude_spectrum:log_energy_band0', 'watch_acceleration:magnitude_autocorrelation:normalized_ac', 'raw_magnet:3d:std_x', 'proc_gyro:magnitude_stats:value_entropy', 'raw_acc:magnitude_stats:moment3', 'watch_acceleration:spectrum:x_log_energy_band1', 'watch_acceleration:magnitude_stats:moment3', 'audio_naive:mfcc11:mean', 'proc_gyro:magnitude_spectrum:spectral_entropy', 'raw_magnet:avr_cosine_similarity_lag_range0', 'audio_naive:mfcc1:std', 'watch_acceleration:spectrum:y_log_energy_band3', 'audio_naive:mfcc2:std', 'raw_magnet:3d:std_y', 'raw_magnet:magnitude_stats:std', 'raw_magnet:3d:ro_xz', 'watch_acceleration:spectrum:x_log_energy_band3', 'watch_acceleration:spectrum:z_log_energy_band2', 'raw_magnet:magnitude_stats:value_entropy', 'location:num_valid_updates', 'audio_naive:mfcc10:mean', 'proc_gyro:3d:ro_xy', 'raw_magnet:avr_cosine_similarity_lag_range4', 'raw_acc:magnitude_stats:value_entropy', 'raw_magnet:avr_cosine_similarity_lag_range1', 'raw_acc:magnitude_autocorrelation:normalized_ac', 'watch_acceleration:spectrum:z_log_energy_band4', 'raw_acc:3d:ro_yz', 'watch_acceleration:spectrum:y_log_energy_band2', 'watch_acceleration:spectrum:x_log_energy_band0', 'raw_magnet:avr_cosine_similarity_lag_range3', 'raw_magnet:magnitude_stats:moment4', 'location:log_latitude_range', 'watch_heading:mean_sin', 'location:best_horizontal_accuracy', 'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4', 'proc_gyro:3d:mean_y', 'proc_gyro:magnitude_spectrum:log_energy_band0', 'watch_heading:mean_cos', 'audio_naive:mfcc3:std', 'raw_magnet:3d:ro_xy', 'watch_acceleration:spectrum:y_log_energy_band1', 'audio_naive:mfcc5:std', 'watch_acceleration:spectrum:z_log_energy_band3', 'location:max_speed', 'proc_gyro:3d:ro_xz', 'raw_magnet:magnitude_spectrum:log_energy_band2', 'location_quick_features:std_lat', 'watch_acceleration:spectrum:z_log_energy_band0', 'proc_gyro:3d:mean_x', 'watch_acceleration:spectrum:z_log_energy_band1', 'proc_gyro:3d:ro_yz', 'audio_naive:mfcc9:std', 'watch_heading:entropy_8bins', 'audio_naive:mfcc12:std', 'lf_measurements:pressure', 'raw_magnet:magnitude_stats:moment3', 'audio_naive:mfcc7:std', 'audio_naive:mfcc8:std', 'watch_heading:std_sin', 'watch_acceleration:magnitude_spectrum:log_energy_band0', 'audio_naive:mfcc11:std', 'proc_gyro:magnitude_spectrum:log_energy_band3', 'raw_magnet:3d:ro_yz', 'raw_acc:magnitude_autocorrelation:period', 'raw_magnet:magnitude_spectrum:log_energy_band1', 'proc_gyro:magnitude_spectrum:log_energy_band1', 'proc_gyro:magnitude_spectrum:log_energy_band2', 'audio_naive:mfcc10:std', 'location:min_speed', 'raw_acc:3d:ro_xz', 'audio_naive:mfcc6:std', 'watch_heading:mom4_sin', 'proc_gyro:3d:mean_z', 'audio_naive:mfcc4:std', 'location:log_longitude_range', 'watch_heading:std_cos', 'raw_magnet:magnitude_spectrum:spectral_entropy', 'raw_acc:3d:ro_xy', 'lf_measurements:light', 'watch_acceleration:magnitude_autocorrelation:period', 'location_quick_features:mean_abs_long_deriv', 'proc_gyro:magnitude_autocorrelation:normalized_ac', 'location_quick_features:mean_abs_lat_deriv', 'watch_heading:mom3_sin', 'raw_magnet:magnitude_autocorrelation:normalized_ac', 'label_source', 'watch_acceleration:3d:ro_xy', 'raw_magnet:magnitude_spectrum:log_energy_band0', 'location_quick_features:long_change', 'watch_acceleration:3d:ro_yz', 'watch_heading:mom3_cos', 'location_quick_features:std_long', 'location:best_vertical_accuracy', 'proc_gyro:magnitude_autocorrelation:period', 'location_quick_features:lat_change', 'watch_heading:mom4_cos', 'raw_magnet:magnitude_autocorrelation:period', 'watch_acceleration:3d:ro_xz', 'watch_acceleration:spectrum:y_log_energy_band0', 'lf_measurements:temperature_ambient', 'lf_measurements:relative_humidity', 'lf_measurements:proximity_cm', 'lf_measurements:proximity']
discrete = discrete[:30]
numerical = numerical[:100]
data_no_disc = data[numerical].to_numpy()
data_with_disc = data[numerical + discrete].to_numpy()
data_labels = data[labels].to_numpy()


# In[ ]:


#print(data_labels)
data_labels_flt = np.argmax(data_labels, axis=1)
#print(data_labels_flt)
plt.hist(data_labels_flt, 8, density = True, 
         histtype ='bar',
         label = labels)
plt.show()
count = pd.Series(data_labels_flt).value_counts()
print ("Element  Count")
print(count)

print(len(data_labels_flt))

no_disc_weighted_f1 = []
no_disc_macro_f1 = []
no_disc_micro_f1 = []
no_disc_acc = []
disc_weighted_f1 = []
disc_macro_f1 = []
disc_micro_f1 = []
disc_acc =[]


# In[ ]:


for i in range(5):
    X_train_no_disc, X_test_no_disc, y_train_no_disc, y_test_no_disc = train_test_split(data_no_disc, data_labels_flt, test_size=0.3)
    scaler = StandardScaler()
    X_train_no_disc = scaler.fit_transform(X_train_no_disc)
    X_test_no_disc = scaler.transform(X_test_no_disc)

    clf_no_disc = MLPClassifier(max_iter=100, hidden_layer_sizes=[40]*int(i/2))
    clf_no_disc.fit(X_train_no_disc, y_train_no_disc)

    y_pred = clf_no_disc.predict(X_test_no_disc)

    print("Simple NN-Based HAR Classifier Trained w/o Discrete Features")
    print(f"Macro F1: {f1_score(y_test_no_disc, y_pred, average='macro')}")
    print(f"Intra-Class F1: {f1_score(y_test_no_disc, y_pred, average='micro')}")
    print(f"Weighted F1: {f1_score(y_test_no_disc, y_pred, average='weighted')}")
    no_disc_weighted_f1.append(round(f1_score(y_test_no_disc, y_pred, average='weighted'),4))
    no_disc_macro_f1.append(round(f1_score(y_test_no_disc, y_pred, average='macro'),4))
    no_disc_micro_f1.append(round(f1_score(y_test_no_disc, y_pred, average='micro'),4))
    no_disc_acc.append(round(accuracy_score(y_test_no_disc, y_pred), 4))


# In[ ]:


for i in range(5):
    X_train_w_disc, X_test_w_disc, y_train_w_disc, y_test_w_disc = train_test_split(data_with_disc, data_labels_flt, test_size=0.3)
    scaler = StandardScaler()
    X_train_w_disc = scaler.fit_transform(X_train_w_disc)
    X_test_w_disc = scaler.transform(X_test_w_disc)

    clf_w_disc = MLPClassifier(max_iter=100, hidden_layer_sizes=[40]*int(i/2))
    clf_w_disc.fit(X_train_w_disc, y_train_w_disc)

    y_pred = clf_w_disc.predict(X_test_w_disc)

    print("Simple NN-Based HAR Classifier Trained w/ Discrete Features")
    print(f"Macro F1: {f1_score(y_test_w_disc, y_pred, average='macro')}")
    print(f"Intra-Class F1: {f1_score(y_test_w_disc, y_pred, average='micro')}")
    print(f"Weighted F1: {f1_score(y_test_w_disc, y_pred, average='weighted')}")
    disc_weighted_f1.append(round(f1_score(y_test_w_disc, y_pred, average='weighted'),4))
    disc_macro_f1.append(round(f1_score(y_test_w_disc, y_pred, average='macro'),4))
    disc_micro_f1.append(round(f1_score(y_test_w_disc, y_pred, average='micro'),4))
    disc_acc.append(round(accuracy_score(y_test_w_disc, y_pred), 4))


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

disc_weighted_f1_avg = round(np.mean(np.array(disc_weighted_f1)),3)
disc_macro_f1_avg = round(np.mean(np.array(disc_macro_f1)),3)
disc_micro_f1_avg = round(np.mean(np.array(disc_micro_f1)),3)
disc_weighted_f1_std = round(np.std(np.array(disc_weighted_f1)),3)
disc_macro_f1_std = round(np.std(np.array(disc_macro_f1)),3)
disc_micro_f1_std = round(np.std(np.array(disc_micro_f1)),3)
disc_acc_avg = round(np.mean(np.array(disc_acc)),3)
disc_acc_std = round(np.std(np.array(disc_acc)),3)

no_disc_weighted_f1_avg = round(np.mean(np.array(no_disc_weighted_f1)),3)
no_disc_macro_f1_avg = round(np.mean(np.array(no_disc_macro_f1)),3)
no_disc_micro_f1_avg = round(np.mean(np.array(no_disc_micro_f1)),3)
no_disc_weighted_f1_std = round(np.std(np.array(no_disc_weighted_f1)),3)
no_disc_macro_f1_std = round(np.std(np.array(no_disc_macro_f1)),3)
no_disc_micro_f1_std = round(np.std(np.array(no_disc_micro_f1)),3)
no_disc_acc_avg = round(np.mean(np.array(no_disc_acc)),3)
no_disc_acc_std = round(np.std(np.array(no_disc_acc)),3)

labels = ['Trained with\nContinuous & Discrete Features', 'Trained with\nContinuous Features']
W_f1 = [disc_weighted_f1_avg, no_disc_weighted_f1_avg]
W_f1_err = [disc_weighted_f1_std, no_disc_weighted_f1_std]
mac_f1 = [disc_macro_f1_avg, no_disc_macro_f1_avg]
mac_f1_err = [disc_macro_f1_std, no_disc_macro_f1_std]
mic_f1 = [disc_micro_f1_avg, no_disc_micro_f1_avg]
mic_f1_err = [disc_micro_f1_std, no_disc_micro_f1_std]
acc = [disc_acc_avg, no_disc_acc_avg]
acc_err = [disc_acc_std, no_disc_acc_std]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, W_f1, width, label='Weighted F1',
                yerr=W_f1_err, alpha=0.5, ecolor='black', capsize=4)
rects2 = ax.bar(x, mac_f1, width, label='Macro F1',
                yerr=mac_f1_err, alpha=0.5, ecolor='black', capsize=4)
rects3 = ax.bar(x + width, acc, width, label='Accuracy',
                yerr=acc_err, alpha=0.5, ecolor='black', capsize=4)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Model Performance')
ax.set_title('HAR Classifier Performance')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
#ax.bar_label(rects4, padding=3)
plt.ylim((0.6, 1.05))
fig.tight_layout()

plt.tight_layout()
plt.savefig('discrete_improvement_w_acc.pdf', dpi=300)  


# In[ ]:


one_hotted = data[labels]
data['labels']=one_hotted.values.argmax(1)

data_sample = data.sample(frac=0.2, random_state=1)
sample_label = data_sample[labels].to_numpy()
sample_labels_flt = np.argmax(sample_label, axis=1)
count = pd.Series(sample_labels_flt).value_counts()
print ("Element  Count")
print(count)

data_train = data.drop(data_sample.index)
new_label = data_train[labels].to_numpy()
new_labels_flt = np.argmax(new_label, axis=1)
count = pd.Series(new_labels_flt).value_counts()
print ("Element  Count")
print(count)

data_train.to_csv("cleaned_data_train.csv")
data_sample.to_csv("cleaned_data_test.csv")

