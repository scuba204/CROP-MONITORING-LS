import ee
import logging
from scripts.gee_functions import DATASETS

# Initialize EE
ee.Initialize(project='winged-tenure-464005-p9')

def test_dataset_integrity(start_date='2024-06-01', end_date='2024-06-30'):
    roi = ee.Geometry.Rectangle([27.5, -30.5, 28.5, -29.5])  # Sample ROI in Lesotho
    broken_datasets = []

    for name, (func, args) in DATASETS.items():
        try:
            logging.info(f"Testing: {name}")
            kwargs = {}
            if 'start' in args: kwargs['start'] = start_date
            if 'end' in args: kwargs['end'] = end_date
            if 'roi' in args: kwargs['roi'] = roi

            result = func(**kwargs)
            if result is None:
                broken_datasets.append((name, "Returned None"))
            else:
                # Check band count
                band_count = len(result.bandNames().getInfo())
                if band_count == 0:
                    broken_datasets.append((name, "No bands"))
                else:
                    logging.info(f"✅ {name} OK - {band_count} bands")
        except Exception as e:
            broken_datasets.append((name, str(e)))
            logging.error(f"❌ {name} failed: {e}")

    # Summary
    print("\n🧪 Dataset Integrity Report:")
    if broken_datasets:
        for name, reason in broken_datasets:
            print(f"❌ {name}: {reason}")
    else:
        print("✅ All datasets loaded successfully!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset_integrity()
