import pandas as pd
import numpy as np
import re
import pprint
from postal.expand import expand_address
import requests
import geocoder

irrelevant_regex = re.compile(r'[^a-z0-9\s]')
multispace_regex = re.compile(r'\s\s+')


df_with_truth = pd.read_csv('restaurant.csv', skip_blank_lines=True)
df_with_truth.head(9)

df = df_with_truth.drop(columns=['cluster', 'phone'])
df.head(9)

def assign_no_symbols_name(df):
    return df.assign(
        name=df['name'].astype(str)
             .str.replace(irrelevant_regex, ' ')
             .str.replace(multispace_regex, ' '))

df = assign_no_symbols_name(df)
df.head(9)

def assign_cleaned_name(df):
    restaurant_stopwords = {
        's',
        'the',
        'la',
        'le',
        'of',
        'and',
        'on',
        'l',
    }
    restaurant_stopwords_regex = r'\b(?:{})\b'.format(
        '|'.join(restaurant_stopwords))
    return df.assign(
        name=df['name'].astype(str)
                       .str.replace(restaurant_stopwords_regex, '')
                       .str.replace(multispace_regex, ' ')
                       .str.strip())

df = assign_cleaned_name(df)
df.head(9)

all_addresses = df['addr'].astype(str).str.cat(
    df['city'].astype(str), sep=', ').values
unique_addresses = np.unique(all_addresses)
print(len(all_addresses), len(unique_addresses))

import os.path
import json

geocoding_filename = 'address_to_geocoding.json'

def geocode_addresses(address_to_geocoding):
    remaining_addresses = set(unique_addresses) - set(address_to_geocoding.keys())
    
    with requests.Session() as session:
        for i, address in enumerate(remaining_addresses):
            print(f"Geocoding {i + 1}/{len(remaining_addresses)}")
            geocode_result = geocoder.google(address, session=session)
            address_to_geocoding[address] = geocode_result.json

        with open(geocoding_filename, 'w') as f:
            json.dump(address_to_geocoding, f, indent=4)

if not os.path.exists(geocoding_filename):
    address_to_geocoding = {}
    geocode_addresses(address_to_geocoding)
else:
    with open(geocoding_filename) as f:
        address_to_geocoding = json.load(f)
    geocode_addresses(address_to_geocoding)
 
address_to_postal = {
    k: v['postal']
    for k, v in address_to_geocoding.items()
    if v is not None and 'postal' in v
}
address_to_latlng = {
    k: (v['lat'], v['lng'])
    for k, v in address_to_geocoding.items()
    if v is not None
}
print(f"Failed to get postal from {len(address_to_geocoding) - len(address_to_postal)}")
print(f"Failed to get latlng from {len(address_to_geocoding) - len(address_to_latlng)}")

def assign_postal_lat_lng(df):
    addresses = df['addr'].astype(str).str.cat(df['city'].astype(str), sep=', ')
    addresses_to_postal = [address_to_postal.get(a) for a in addresses]
    addresses_to_lat = [address_to_latlng[a][0] if a in address_to_latlng else None for a in addresses]
    addresses_to_lng = [address_to_latlng[a][1] if a in address_to_latlng else None for a in addresses]

    return df.assign(postal=addresses_to_postal, lat=addresses_to_lat, lng=addresses_to_lng)

df = assign_postal_lat_lng(df)
df.head(9)

def assign_addr_variations(df):
    return df.assign(
        addr_variations=df['addr'].astype(str).apply(
            lambda addr: frozenset(expand_address(addr))))

df = assign_addr_variations(df)
df.head(9)

df_training = pd.read_csv('restaurant-training.csv', skip_blank_lines=True)
df_training

df = assign_no_symbols_name(df)
df = assign_cleaned_name(df)
df = assign_postal_lat_lng(df)
df = assign_addr_variations(df)
df.head(4)

import dedupe

def addr_variations_comparator(x, y):
    return 1 - int(bool(x.intersection(y)))

fields = [
    {
        'field': 'name',
        'variable name': 'name',
        'type': 'ShortString',
        'has missing': True
    },
    {
        'field': 'addr',
        'variable name': 'addr',
        'type': 'ShortString',
    },
    {
        'field': 'city',
        'variable name': 'city',
        'type': 'ShortString',
        'has missing': True
    },
    {
        'field': 'postal',
        'variable name': 'postal',
        'type': 'ShortString',
        'has missing': True
    },
    {
        'field': 'latlng',
        'variable name': 'latlng',
        'type': 'LatLong',
        'has missing': True
    },
    {
        'field': 'addr_variations',
        'variable name': 'addr_variations',
        'type': 'Custom',
        'comparator': addr_variations_comparator
    },
    {
        'type': 'Interaction',
        'interaction variables': [
            'addr',
            'city',
            'postal',
            'latlng',
            'addr_variations'
        ]
    }
]

settings_filename = 'dedupe-settings.pickle'
if os.path.exists(settings_filename):
    with open(settings_filename, 'rb') as sf:
        deduper = dedupe.StaticDedupe(sf, num_cores=4)
else:
    deduper = dedupe.Dedupe(fields, num_cores=4)

data_for_dedupe = df.to_dict('index')
for record in data_for_dedupe.values():
    for k, v in record.items():
        if isinstance(v, float) and np.isnan(v):
            record[k] = None
    
    lat = record.pop('lat')
    lng = record.pop('lng')
    if lat is not None and lng is not None:
        record['latlng'] = (lat, lng)
    else:
        record['latlng'] = None

if not isinstance(deduper, dedupe.StaticDedupe):
    deduper.sample(data_for_dedupe)
    
    training_filename = 'dedupe-slides-training.json'
    if os.path.exists(training_filename):
        with open(training_filename) as tf:
            deduper.readTraining(tf)

    dedupe.consoleLabel(deduper)

threshold = deduper.threshold(data_for_dedupe, recall_weight=2)
clustered_dupes = deduper.match(data_for_dedupe, threshold)