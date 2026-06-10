import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Random_Address import generate_random_address


def generate_patient_data(n_patients=1000, lambda_per_hour=15, start_time=None):
    """
    Generate arrival times and addresses for patients based on a Poisson process.
    
    Parameters:
    -----------
    n_patients : int
        Number of patients to generate (default: 1000)
    lambda_per_hour : float
        Average arrival rate per hour (default: 15)
    start_time : datetime
        Start time for the first patient (default: today at 00:00)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: patient_id, arrival_time, address, latitude, longitude
    """
    
    # Set default start time
    if start_time is None:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Convert lambda from per-hour to per-minute
    lambda_per_minute = lambda_per_hour / 60
    
    # Generate inter-arrival times from exponential distribution
    # For a Poisson process with rate lambda, inter-arrival times follow Exp(lambda)
    inter_arrival_times = np.random.exponential(scale=1/lambda_per_minute, size=n_patients)
    
    # Convert to cumulative arrival times in minutes
    arrival_times_minutes = np.cumsum(inter_arrival_times)
    
    # Convert to datetime objects
    arrival_times = [start_time + timedelta(minutes=t) for t in arrival_times_minutes]
    
    # Generate patient data
    patients = []
    
    for patient_id in range(1, n_patients + 1):
        # Generate address using the shared function
        address_data = generate_random_address(max_attempts=10)
        
        if address_data:
            address = address_data['address']
            latitude = address_data['latitude']
            longitude = address_data['longitude']
        else:
            # Fallback if geocoding fails
            address = "Address not available"
            latitude = None
            longitude = None
        
        patients.append({
            'patient_id': patient_id,
            'arrival_time': arrival_times[patient_id - 1],
            'address': address,
            'latitude': latitude,
            'longitude': longitude
        })
    
    # Create DataFrame
    df = pd.DataFrame(patients)
    
    return df


if __name__ == "__main__":
    # Generate patient data for 100 patients with lambda=15 per hour
    print("Generating patient data for 100 patients...")
    patient_df = generate_patient_data(n_patients=100, lambda_per_hour=15)
    
    # Display summary statistics
    print("\n" + "="*80)
    print("Patient Data Summary")
    print("="*80)
    print(f"Total patients: {len(patient_df)}")
    print(f"First patient arrival: {patient_df['arrival_time'].min()}")
    print(f"Last patient arrival: {patient_df['arrival_time'].max()}")
    print(f"Total duration: {patient_df['arrival_time'].max() - patient_df['arrival_time'].min()}")
    print(f"\nFirst 10 patients:")
    print(patient_df.head(10).to_string())
    
    # Save to CSV
    output_file = "Patient_Data_Generated.csv"
    patient_df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to {output_file}")
