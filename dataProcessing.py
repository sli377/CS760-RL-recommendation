# Reinforcement Learning-Oriented Data Preparation Script for Yelp Dataset
# This builds on top of large-scale data processing principles (Lee et al., 2022) and prepares temporal interaction sequences for RL-based recommender systems

import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm  # for progress bar

# Step 0: Define paths and constants
data_path = r"C:\\Users\\lsy135\\Desktop\\760\\archive"
business_path = os.path.join(data_path, 'yelp_academic_dataset_business.json')
review_path = os.path.join(data_path, 'yelp_academic_dataset_review.json')
user_path = os.path.join(data_path, 'yelp_academic_dataset_user.json')
checkin_path = os.path.join(data_path, 'yelp_academic_dataset_checkin.json')

WINDOW_SIZE = 5  # Used for building state-action sequences
DECAY_LAMBDA = 0.1  # Decay factor for recency weighting

# Step 1: Load business categories for each business_id
business_categories = {}
with open(business_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Loading business categories"):
        try:
            business = json.loads(line)
            bid = business.get('business_id')
            categories = business.get('categories', '')
            if bid and categories:
                category_list = [c.strip() for c in categories.split(',') if c.strip()]
                business_categories[bid] = category_list[:5]  # take top 5 categories
        except:
            continue

# Clean and fill missing values in a single entry

def clean_entry(user_id, business_id, stars, date, user_profiles, business_checkin_popularity):
    if not date or pd.isnull(date):
        return None
    hour = date.hour
    if 5 <= hour < 12:
        time_segment = 'morning'
    elif 12 <= hour < 18:
        time_segment = 'afternoon'
    else:
        time_segment = 'evening'

    try:
        stars = float(stars)
    except:
        stars = 0.0

    return {
        'business_id': business_id or 'unknown',
        'stars': stars,
        'date': date.strftime('%Y-%m-%d %H:%M:%S'),
        'parsed_date': date,
        'time_segment': time_segment,
        'user_profile': user_profiles.get(user_id, {'review_count': 0, 'average_stars': 0.0, 'friend_count': 0}),
        'business_checkin': business_checkin_popularity.get(business_id, {'morning': 0, 'afternoon': 0, 'evening': 0}),
        'business_categories': business_categories.get(business_id, [])
    }

# Step 2: Load user profile data into memory
user_profiles = {}
with open(user_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Loading user profiles"):
        try:
            user = json.loads(line)
            user_id = user.get('user_id')
            if not user_id:
                continue
            profile = {
                'review_count': user.get('review_count', 0),
                'average_stars': user.get('average_stars', 0.0),
                'friend_count': len(user.get('friends', '').split(', ')) if user.get('friends') else 0
            }
            user_profiles[user_id] = profile
        except json.JSONDecodeError:
            continue

# Step 3: Load check-in data and calculate business time-of-day popularity
business_checkin_popularity = {}
with open(checkin_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Processing check-ins"):
        try:
            item = json.loads(line)
            business_id = item.get('business_id')
            date_strs = item.get('date', '')
            if not business_id or not date_strs:
                continue
            times = date_strs.split(', ')
            count_morning = count_afternoon = count_evening = 0
            for t in times:
                dt = pd.to_datetime(t, errors='coerce')
                if pd.isnull(dt):
                    continue
                hour = dt.hour
                if 5 <= hour < 12:
                    count_morning += 1
                elif 12 <= hour < 18:
                    count_afternoon += 1
                else:
                    count_evening += 1
            business_checkin_popularity[business_id] = {
                'morning': count_morning,
                'afternoon': count_afternoon,
                'evening': count_evening
            }
        except json.JSONDecodeError:
            continue

# Step 4: Stream reviews using line-by-line JSON parsing (JSONL format)
user_sequences = {}
with open(review_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Parsing reviews"):
        try:
            review = json.loads(line)
            user_id = review.get('user_id')
            business_id = review.get('business_id')
            stars = review.get('stars')
            date_str = review.get('date')
            if not (user_id and business_id and stars and date_str):
                continue
            date = pd.to_datetime(date_str, errors='coerce')
            entry = clean_entry(user_id, business_id, stars, date, user_profiles, business_checkin_popularity)
            if entry is None:
                continue
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append(entry)
        except Exception:
            continue

# Step 5: Sort user interactions by time
for uid in tqdm(user_sequences, desc="Sorting interactions"):
    user_sequences[uid] = sorted(user_sequences[uid], key=lambda x: x['parsed_date'])

# Step 6: Generate RL training samples with sliding window + time decay features
rl_samples = []
for user_id, interactions in tqdm(user_sequences.items(), desc="Generating RL samples"):
    if len(interactions) <= WINDOW_SIZE:
        continue
    for i in range(WINDOW_SIZE, len(interactions)):
        state = interactions[i - WINDOW_SIZE:i]
        action = interactions[i]
        next_state = interactions[i - WINDOW_SIZE + 1:i + 1]
        reward = 1 if action['stars'] >= 4 else 0

        # Add time-based decay feature to state
        action_time = pd.to_datetime(action['date'])
        for s in state:
            if 'parsed_date' in s:
                s['days_since_action'] = max((action_time - s['parsed_date']).days, 0)
                s['recency_weight'] = round(np.exp(-DECAY_LAMBDA * s['days_since_action']), 6)
                s.pop('parsed_date', None)

        for s in next_state:
            s.pop('parsed_date', None)  # also clean next_state

        rl_samples.append({
            'user_id': user_id,
            'state': state,
            'action': action['business_id'],
            'action_time_segment': action['time_segment'],
            'user_profile': action['user_profile'],
            'business_checkin': action['business_checkin'],
            'business_categories': action['business_categories'],
            'reward': reward,
            'next_state': next_state
        })


# Step 7: Save to disk
output_file = r"C:\\Users\\lsy135\\Desktop\\760\\rl_dataset.json"
chunk_size = 10000
with open(output_file, 'w', encoding='utf-8') as f:
    for i in tqdm(range(0, len(rl_samples), chunk_size), desc="Writing output"):
        chunk = rl_samples[i:i+chunk_size]
        json.dump(chunk, f, indent=2)
        f.write('\n')

print(f"Processed {len(rl_samples)} RL samples and saved to {output_file}")
