import pandas as pd
import json
import os
import numpy as np

# Define paths to the JSON files - change to your path
data_path = r"C:\Users\lsy135\Desktop\760\archive"

business_path = os.path.join(data_path, 'yelp_academic_dataset_business.json')
review_path = os.path.join(data_path, 'yelp_academic_dataset_review.json')
user_path = os.path.join(data_path, 'yelp_academic_dataset_user.json')
tip_path = os.path.join(data_path, 'yelp_academic_dataset_tip.json')
checkin_path = os.path.join(data_path, 'yelp_academic_dataset_checkin.json')

# Load JSON files as pandas DataFrames
def load_json(file_path, num_lines=100000):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(data) >= num_lines:
                break
    return pd.DataFrame(data)

# Load datasets
business_df = load_json(business_path)
review_df = load_json(review_path)
user_df = load_json(user_path)
tip_df = load_json(tip_path)
checkin_df = load_json(checkin_path)

# Preprocessing and Cleaning
business_df = business_df[['business_id', 'name', 'categories', 'attributes', 'stars', 'review_count', 'is_open',
                           'address', 'city', 'state', 'postal_code']]
business_df['categories'] = business_df['categories'].fillna('').apply(lambda x: x.split(', ') if x else [])

review_df = review_df[['review_id', 'user_id', 'business_id', 'stars', 'text', 'date']]
review_df['date'] = pd.to_datetime(review_df['date'], errors='coerce')
review_df = review_df.sort_values(by=['user_id', 'date'])

user_df = user_df[['user_id', 'name', 'review_count', 'average_stars', 'friends', 'yelping_since']]
user_df['friends'] = user_df['friends'].apply(lambda x: x.split(', ') if isinstance(x, str) and x != 'None' else [])
user_df['yelping_since'] = pd.to_datetime(user_df['yelping_since'], errors='coerce')

tip_df = tip_df[['user_id', 'business_id', 'text', 'date']]
tip_df['date'] = pd.to_datetime(tip_df['date'], errors='coerce')

checkin_df['date'] = checkin_df['date'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
checkin_df = checkin_df.explode('date')
checkin_df['date'] = pd.to_datetime(checkin_df['date'], errors='coerce')

# Merge Reviews with Business Info
review_merged = pd.merge(review_df, business_df, on='business_id', how='left')
review_merged = pd.merge(review_merged, user_df, on='user_id', how='left')

interaction_df = pd.concat([review_merged, tip_df], ignore_index=True)
checkin_merged = pd.merge(checkin_df, business_df, on='business_id', how='left')
interaction_df = pd.concat([interaction_df, checkin_merged], ignore_index=True)

# Sort interactions by user and time for sequential modeling
interaction_df = interaction_df.sort_values(by=['user_id', 'date'])

# Replace all NaT, NaN, or None values with Python `None` type for compatibility
interaction_df = interaction_df.replace({pd.NaT: None, np.nan: None})

# Convert all datetime objects to strings
def convert_to_string(x):
    if isinstance(x, pd.Timestamp):
        return x.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(x, list) or isinstance(x, np.ndarray):
        return [str(item) if isinstance(item, pd.Timestamp) else item for item in x]  # Convert timestamps within lists
    elif x is None or pd.isna(x):  # Handles NaT, np.nan, or None
        return None

    else:
        return str(x) if isinstance(x, (int, float, str, bool)) else x  # Convert standard objects to strings

# Apply the conversion function to all columns
for col in interaction_df.columns:
    interaction_df[col] = interaction_df[col].apply(convert_to_string)

# Checking if there are still Timestamp objects in the DataFrame
timestamp_columns = [col for col in interaction_df.columns if interaction_df[col].apply(lambda x: isinstance(x, pd.Timestamp)).any()]
if timestamp_columns:
    print(f"Error: Some columns still contain Timestamp objects: {timestamp_columns}")
else:
    print("All Timestamps successfully converted to strings.")

# Convert entire DataFrame to dictionary-friendly format
interaction_dict = interaction_df.to_dict(orient='records')

# Creating User-Item Interaction Sequences
user_sequences = {}
for row in interaction_dict:
    user_id = row.get('user_id')
    if user_id not in user_sequences:
        user_sequences[user_id] = []
    user_sequences[user_id].append(row)

# Save user sequences to a JSON file for easy access later
with open('user_sequences.json', 'w', encoding='utf-8') as f:
    json.dump(user_sequences, f)

print(f"Saved user sequences for {len(user_sequences)} users.")


# Remove redundant columns from the interaction DataFrame
interaction_df_cleaned = interaction_df.drop(columns=[
    'name_y', 'review_count_y', 'average_stars', 'friends', 'yelping_since', 'name', 'stars', 'review_count'
])

# Rename relevant columns for better clarity
interaction_df_cleaned = interaction_df_cleaned.rename(columns={
    'stars_x': 'user_rating',
    'stars_y': 'business_average_rating',
    'name_x': 'business_name',
    'review_count_x': 'business_review_count'
})

# Replace null values with default values or drop rows if necessary
interaction_df_cleaned = interaction_df_cleaned.fillna({
    'business_average_rating': 0,
    'business_review_count': 0,
    'user_rating': 0,
    'address': 'Unknown',
    'city': 'Unknown',
    'state': 'Unknown',
    'postal_code': 'Unknown'
})

# Print sample cleaned data
# Display a full sample row from interaction_df_cleaned
sample_index = 10  # Choose an index for the sample you want to display

# Retrieve the sample row
sample_row = interaction_df_cleaned.iloc[sample_index].to_dict()

# Pretty-print the sample row for better readability
print(json.dumps(sample_row, indent=4))

# Save the merged interaction DataFrame to a JSON file - change to your path
output_path_json = r"C:\Users\lsy135\Desktop\760\merged_interaction_cleaned_data.json"
chunk_size = 10000  # Adjust based on your memory capacity
total_rows = len(interaction_df_cleaned)

with open(output_path_json, 'w', encoding='utf-8') as f:
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = interaction_df_cleaned.iloc[start:end]

        # Convert chunk to JSON string
        json_str = chunk.to_json(orient='records', lines=True)

        # Write chunk to file
        f.write(json_str + '\n')  # Add newline after each chunk for readability

print(f"Data saved successfully to {output_path_json} in chunks.")
