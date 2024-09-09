import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.preprocessing import LabelBinarizer # type: ignore
import streamlit as st  # type: ignore
from streamlit_option_menu import option_menu # type: ignore
import re
import pickle as pk

st.set_page_config(page_title="Industrial Copper Modelling",
                   layout="wide",
                   initial_sidebar_state ="auto",
                   menu_items={'About': "This was done by Narmadha Devi B"})



with st.sidebar:
    selected = option_menu(None, ["Home","Selling Price Prediction","Status Prediction"],
                           default_index=0,
                           orientation="vertical",
                           styles = {"nav-link-selected": {"background-color": "#008000"}})

#st.title(':green[Industrial Copper Modelling]')

if selected == "Home":
    st.title(':green[Industrial Copper Modelling]')
    st.subheader(':blue[Domain:] Manufacturing')
    st.subheader(':blue[Overview:] Create a Streamlit application to show there predicted  status (WON/LOST) and selling price using machine learning models. Incorporate data cleaning, preprocessing, and EDA to prepare your data. The app will let users input data, apply preprocessing, and get predictions for both regression and classification tasks.')
    st.subheader(':blue[Skills Take Away:] Python Scripting, Data Preprocessing, EDA (Exploratory Data Analysis) , Streamlit')

if selected == "Selling Price Prediction":
    st.title(':green[Selling Price Prediction]')
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

         
    with st.form("my_form"):
            col1,col2=st.columns([5,5])
            with col1:
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
            with col2:
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")

            flag=0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:
                if re.match(pattern, i):
                    pass
                else:
                    flag=1
                    break

    if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)

    if submit_button and flag==0:

            import pickle as pk 
            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/model.pkl", 'rb') as file:
                load_model = pk.load(file)

            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/scalar.pkl", 'rb') as file:
                load_scalar = pk.load(file)

            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/type.pkl", 'rb') as file:
                load_type = pk.load(file) 

            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/status.pkl", 'rb') as file:
                load_status = pk.load(file) 

            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            new_sample_type = load_type.transform(new_sample[:, [7]]).toarray()
            new_sample_status = load_status.transform(new_sample[:, [8]]).toarray()
            new_sample_numrical = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_type, new_sample_status), axis=1)
            new_sample_scalar = load_scalar.transform(new_sample_numrical)
            new_pred = load_model.predict(new_sample_scalar)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))



if selected == "Status Prediction":

    st.title(':green[Status Prediction]')

    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    with st.form("my_form1"):
            col1,col2=st.columns([5, 5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

            with col2:
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)
                cproduct_ref = st.selectbox("Product Reference", product,key=51)
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")

            cflag=0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:
                if re.match(pattern, k):
                    pass
                else:
                    cflag=1
                    break

    if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)

    if csubmit_button and cflag==0:
            import pickle as pk
            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/Classific-model.pkl", 'rb') as file:
                load_cmodel = pk.load(file)

            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/Classific-scalar.pkl", 'rb') as file:
                load_cscalar = pk.load(file)

            with open("C:/Users/Bala Krishnan/OneDrive/Desktop/Industrial Copper Modelling/Classific-type.pkl", 'rb') as file:
                load_ctype = pk.load(file)

            # Predict the status for a new sample
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(cproduct_ref),citem_type]])
            new_sample_ctype = load_ctype.transform(new_sample[:, [8]]).toarray()
            new_sample_cnumerical = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ctype), axis=1)
            new_sample_cscalar = load_cscalar.transform(new_sample_cnumerical)
            new_pred = load_cmodel.predict(new_sample_cscalar)
            if new_pred==1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
