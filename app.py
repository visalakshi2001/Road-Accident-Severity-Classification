import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from Prediction import get_prediction, LabelEncoder
from load_model import get_model

rf_model = get_model(model_path = r'Model/RTA_severity model.pkl')


st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu

options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']

options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown'] 

options_Road_allignment = ['Tangent road with flat terrain', 
       'Tangent road with mild grade and flat terrain', 'Escarpments',
       'Tangent road with rolling terrain', 'Gentle horizontal curve',
       'Tangent road with mountainous terrain and',
       'Steep grade downward with mountainous terrain',
       'Sharp reverse curve',
       'Steep grade upward with mountainous terrain']

options_Types_of_Junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other',
       'Unknown', 'T Shape', 'X Shape']

options_Road_surface_conditions = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']

options_Light_conditions = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
       'Darkness - lights unlit']
options_Weather_conditions = ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other',
       'Windy', 'Snow', 'Unknown', 'Fog or mist']
options_Type_of_collision = ['Collision with roadside-parked vehicles',
       'Vehicle with vehicle collision',
       'Collision with roadside objects', 'Collision with animals',
       'Other', 'Rollover', 'Fall from vehicles',
       'Collision with pedestrians', 'With Train', 'Unknown']
options_Number_of_vehicles_involved = [2, 1, 3, 6, 4, 7]

options_Pedestrian_movement = ['Not a Pedestrian', "Crossing from driver's nearside",
       'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
       'Unknown or other',
       'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
       'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
       'Walking along in carriageway, back to traffic',
       'Walking along in carriageway, facing traffic',
       'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle']

options_Age_band_of_casualty = ['na', '31-50', '18-30', 'Under 18', 'Over 51', '5']



features = ['hour','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        casualties = st.slider("Hour of Accident: ", 1, 8, value=0, format="%d")
        accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.slider("Pickup Hour: ", 1, 7, value=0, format="%d")
        vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        Road_allignment = st.selectbox("Select Road allignment: ", options=options_Road_allignment)
        Types_of_Junction = st.selectbox("Select Type of junction: ", options=options_Types_of_Junction)
        Road_surface_conditions = st.selectbox("Select Road condition: ", options=options_Road_surface_conditions)
        Light_conditions = st.selectbox("Select Light condition: ", options=options_Light_conditions)
        Weather_conditions = st.selectbox("Select weather condition: ", options=options_Weather_conditions)
        Type_of_collision = st.selectbox("Select type of collision: ", options=options_Type_of_collision)
        Number_of_vehicles_involved = st.slider("Number of vehicles involved: ", 1, 8, value=2, format="%d")
        Pedestrian_movement = st.selectbox("Select pedestrian movement: ", options=options_Pedestrian_movement)
        Age_band_of_casualty = st.slider("Age band of casualty: ", 0, 60, value=30, format="%d")
        submit = st.form_submit_button("Predict")



    if submit:
        
        accident_cause = LabelEncoder(accident_cause, options_cause)
        vehicle_type = LabelEncoder(vehicle_type, options_vehicle_type)
        driver_age =  LabelEncoder(driver_age, options_age)
        accident_area =  LabelEncoder(accident_area, options_acc_area)
        driving_experience = LabelEncoder(driving_experience, options_driver_exp) 
        lanes = LabelEncoder(lanes, options_lanes)
        Road_allignment = LabelEncoder(Road_allignment, options_Road_allignment) 
        Types_of_Junction = LabelEncoder(Types_of_Junction, options_Types_of_Junction) 
        Road_surface_conditions = LabelEncoder(Road_surface_conditions, options_Road_surface_conditions) 
        Light_conditions = LabelEncoder(Light_conditions, options_Light_conditions) 
        Weather_conditions = LabelEncoder(Weather_conditions, options_Weather_conditions) 
        Type_of_collision = LabelEncoder(Type_of_collision, options_Type_of_collision) 
        Pedestrian_movement = LabelEncoder(Pedestrian_movement, options_Pedestrian_movement) 
        


        data = np.array([hour,casualties,accident_cause,vehicles_involved, 
                            vehicle_type,driver_age,accident_area,driving_experience,
                            lanes,Road_allignment,Types_of_Junction,Road_surface_conditions,
                            Light_conditions,Weather_conditions,Type_of_collision,Number_of_vehicles_involved,
                            Pedestrian_movement,Age_band_of_casualty
                            ]).reshape(1,-1)
        print(data)
        pred = get_prediction(data=data, model=rf_model)

        st.write(f"The predicted severity is:  {pred[0]}")


if __name__ == '__main__':
    main()
