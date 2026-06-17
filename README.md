# Planning-Scheduling

Install necessary requirements:

pip install geopy 
pip install streamlit



Patient_Generator.py generates the patients and saves this in patients.csv

providers.csv contains the info of the home care organisations (providers)

Rolling_Horizon_Allocation.py loads the csv files of the patients and providers and assigns the patients to the providers using a rolling horizon algorithm. 