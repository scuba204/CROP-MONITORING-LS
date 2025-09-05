# faostat_preprocess.py
import pandas as pd
import numpy as np

# Input file path (update if your file is stored somewhere else)
INPUT_FILE = 'data/FAOSTAT_data_en_8-19-2025.csv'

# Output file names
OUTPUT_YIELD = "faostat_lesotho_yield_only.csv"
OUTPUT_AREA = "faostat_lesotho_area_shares.csv"
OUTPUT_FEATURES = "faostat_rotation_proxy_features.csv"

def main():
    # Load raw data
    df = pd.read_csv(INPUT_FILE)

    # Ensure numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # --- YIELD DATASET ---
    yield_df = df[df['Element'].str.lower() == 'yield'].copy()
    yield_df = yield_df[['Area','Item','Year','Unit','Value','Flag','Flag Description']]
    yield_df.rename(columns={'Value':'Yield','Unit':'Yield_Unit'}, inplace=True)

    # Save yield-only table
    yield_df.to_csv(OUTPUT_YIELD, index=False)
    print(f"✅ Saved yield-only file -> {OUTPUT_YIELD}")

    # --- AREA HARVESTED DATASET ---
    area_df = df[df['Element'].str.lower() == 'area harvested'].copy()
    area_df = area_df[['Area','Item','Year','Unit','Value','Flag','Flag Description']]
    area_df.rename(columns={'Value':'Area_ha','Unit':'Area_Unit'}, inplace=True)

    # Compute total area per year
    total_area_per_year = area_df.groupby('Year')['Area_ha'].sum().rename('total_area')
    area_yr = area_df.groupby(['Year','Item'])['Area_ha'].sum().reset_index()
    area_yr = area_yr.merge(total_area_per_year, on='Year', how='left')
    area_yr['area_share'] = area_yr['Area_ha'] / area_yr['total_area']

    # Lag previous-year share (rotation proxy)
    area_yr['prev_area_share'] = (
        area_yr.sort_values(['Item','Year'])
        .groupby('Item')['area_share']
        .shift(1)
    )

    # Save area shares
    area_yr.to_csv(OUTPUT_AREA, index=False)
    print(f"✅ Saved area shares file -> {OUTPUT_AREA}")

    # --- FEATURES PROTOTYPE ---
    features_proto = yield_df.merge(
        area_yr[['Year','Item','prev_area_share']],
        on=['Year','Item'],
        how='left'
    )
    features_proto = features_proto.dropna(subset=['Yield'])

    features_proto.to_csv(OUTPUT_FEATURES, index=False)
    print(f"✅ Saved prototype features file -> {OUTPUT_FEATURES}")

    print("\nAll done! You now have three files ready for modeling.")

if __name__ == "__main__":
    main()
