from datasets import load_dataset

DATASET = "biglam/blbooks-parquet"

print("Dataset:", DATASET)

dataset = load_dataset(DATASET, split='train', streaming=True, trust_remote_code=True)

print(dataset)

seventeenth_century = []
eighteenth_century = []
nineteenth_century = []

# Get first 1000 samples of dataset
for _ in range(1000):
    sample = next(iter(dataset))
    print(sample['raw_date'])
    year = int(sample['raw_date'])