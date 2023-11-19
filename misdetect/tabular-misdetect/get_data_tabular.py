import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import copy


columns = {"eeg": ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Eye'],
           "uscensus": ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num', 'Marital-status', 'Occupation',
                    'Relationship', 'Race', 'Sex', 'Capital-gain', ' Capital-loss', 'Hours-per-week', 'Native-country', 'Income'],
           "credit": ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                    'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'],
           "hotel": ['Booking_ID', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
                    'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year',
                    'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
                    'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests', 'booking_status'],
           "heart": ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR',
                    'ExerciseAngina','Oldpeak','ST_Slope','HeartDisease'],
           "wine": ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide',
                    'total_sulfur_dioxide','density','pH','sulphates','alcohol','quality','color'],
           "airline": ['id','Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service',
                    'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding',
                    'Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service',
                    'Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes','satisfaction'],
           "mobile": [ 'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
                    'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi', 'price_range'],
           "covertype": ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                    'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
                    'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                    'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
                    'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
                    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
                    'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type'],
           }

classes = {"eeg": ['0', '1'],
           "uscensus": ['0', '1'],
           "credit": ['0', '1'],
           "hotel": ['0', '1'],
           "heart": ['0', '1'],
           "wine": ['0', '1', '2', '3', '4', '5', '6'],
           "airline": ['0', '1'],
           "mobile": ['0', '1', '2', '3'],
           "covertype": ['0', '1', '2', '3', '4', '5', '6'],
           }

labels = {  "eeg": "Eye",
           "uscensus": "Income",
           "credit": "SeriousDlqin2yrs",
           "hotel": "booking_status",
           "heart": "HeartDisease",
           "wine": "quality",
           "airline": "satisfaction",
           "mobile": "price_range",
           "covertype": "Cover_Type",
           }

def mis_injection(dataset, mis_rate, mis_distribution):
    label_index = columns[dataset].index(labels[dataset])

    # clean数据占比
    good_sample_ratio = 1 - float(mis_rate)
    train_set_before = pd.read_csv("../../dataset/" + dataset + "/" + dataset + "_normalize.csv")
    if dataset in ["credit", "airline"]:
        train_set_before = train_set_before.fillna(axis=1, method='ffill')
        train_set_before = train_set_before.dropna()
        # 首先将pandas读取的数据转化为array
        train_set_before = np.array(train_set_before)
        train_set_before = np.delete(train_set_before, 0, axis=1)     

    train_set_before = np.array(train_set_before)
    label = np.array(train_set_before).T[label_index]
    train_set_before = np.delete(train_set_before, label_index, axis=1)

    # 将train_set转化为特定list形式
    train_set = []
    for i in range(len(train_set_before)):
        tmp = [torch.tensor(train_set_before[i].tolist()), int(label[i])]
        train_set.append(tmp)

    train_set_tmp = copy.deepcopy(train_set)

    if mis_distribution == "random":
        cnt_label = {}
        for idx, tensor in enumerate(train_set):
            cnt_label[tensor[1]] = cnt_label.get(tensor[1], 0) + 1
        print(len(cnt_label))

        cnt_good_label_tgt = {}
        for k, v in cnt_label.items():
            cnt_good_label_tgt[k] = int(v * good_sample_ratio)

        manipulate_label = {}
        good_idx_set = []
        for idx, tensor in enumerate(train_set):
            manipulate_label[tensor[1]] = manipulate_label.get(tensor[1], 0) + 1
            if manipulate_label[tensor[1]] > cnt_good_label_tgt[tensor[1]]:
                p = np.random.randint(0, len(cnt_label))
                while True:
                    if p != tensor[1]:
                        train_set[idx][1] = p
                        break
                    p = np.random.randint(0, len(cnt_label))
            else:
                good_idx_set.append(idx)

        good_idx_array = np.array(good_idx_set)
        all_idx_array = np.arange(len(train_set))
        bad_idx_array = np.setdiff1d(all_idx_array, good_idx_array)
        train_clean_dataset = []
        for i in good_idx_array:
            train_clean_dataset.append(train_set[i])
            if train_set[i][1] != train_set_tmp[i][1]:
                print("--------------------------------")
        train_bad_dataset = []
        for i in bad_idx_array:
            train_bad_dataset.append(train_set[i])
            if train_set[i][1] == train_set_tmp[i][1]:
                print("--------------------------------")

        train_bad_dataset = []
        for i in bad_idx_array:
            train_bad_dataset.append(train_set_tmp[i])

        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset

        train_clean_bad_set = train_clean_dataset + train_bad_dataset
        print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
        return train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth

    else:
        # ---------------------------------------------------------------
        # 随机制造脏数据，而不是每个类取固定比例制造脏数据
        train_clean_size = int(good_sample_ratio * len(train_set_tmp))
        train_bad_size = len(train_set_tmp) - train_clean_size
        train_clean_set, train_bad_set = torch.utils.data.random_split(train_set_tmp, [train_clean_size, train_bad_size])
        
        train_set = []
        for i in train_set_tmp:
            train_set.append(list(i))
        
        train_clean_dataset = []
        train_bad_dataset = []
        for i in train_bad_set:
            train_bad_dataset.append(list(i))
        
        for i in train_clean_set:
            train_clean_dataset.append(list(i))
        
        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset
        
        for i in train_bad_dataset:
            p = np.random.randint(0, len(classes))
            while True:
                if p != i[1]:
                    i[1] = p
                    break
                p = np.random.randint(0, len(classes))
        
        train_clean_bad_set_ground_truth = train_clean_dataset + train_bad_dataset
        train_clean_bad_set = train_clean_dataset + train_bad_dataset
        print(len(train_clean_dataset), len(train_bad_dataset), len(train_clean_bad_set))
        return train_clean_dataset, train_bad_dataset, train_clean_bad_set, train_clean_bad_set_ground_truth
