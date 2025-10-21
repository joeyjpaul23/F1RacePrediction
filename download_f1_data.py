import os
import fastf1
import pandas as pd

# Ensure cache and data directories exist
os.makedirs('f1_cache', exist_ok=True)
os.makedirs('data', exist_ok=True)

fastf1.Cache.enable_cache('f1_cache')

for year in [2019]:
    all_merged = []
    print(f"\nProcessing year: {year}")
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Could not get schedule for {year}: {e}")
        continue

    for _, race in schedule.iterrows():
        try:
            event = fastf1.get_event(year, race['RoundNumber'])

            # Get race results
            race_session = event.get_race()
            race_session.load()
            race_results = race_session.results
            race_results = race_results.rename(columns=lambda x: f"Race_{x}")

            # Get qualifying results
            qual_session = event.get_qualifying()
            qual_session.load()
            qual_results = qual_session.results
            qual_results = qual_results.rename(columns=lambda x: f"Qual_{x}")

            # Merge on driver number (which is unique for each driver in a season)
            merged = pd.merge(
                race_results,
                qual_results,
                left_on="Race_DriverNumber",
                right_on="Qual_DriverNumber",
                suffixes=('_race', '_qual'),
                how='left'
            )

            merged['Year'] = year
            merged['Round'] = race['RoundNumber']
            merged['EventName'] = race['EventName']
            all_merged.append(merged)
            print(f"  ✓ Merged race and qualifying for {race['EventName']} {year}")

        except Exception as e:
            print(f"  ✗ Could not process {race['EventName']} {year}: {e}")

    if all_merged:
        df = pd.concat(all_merged, ignore_index=True)
        df.to_csv(f"data/f1_results_{year}.csv", index=False)
        print(f"Saved data/f1_results_{year}.csv")
    else:
        print(f"No combined data for {year}")