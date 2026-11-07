from datasets import load_dataset

DATASET = "biglam/blbooks-parquet"

print("Dataset:", DATASET)

dataset = load_dataset(DATASET, split='train', streaming=True, trust_remote_code=True)

print(dataset)

seventeenth_century = []
eighteenth_century = []
nineteenth_century = []
seen_authors = set()
seen_all_names = set()
seen_ids = set()

# Get 50 valid book excerpts for each century
for i in range(100000):
    sample = next(iter(dataset))

    if i == 0:
        print("First Sample: ", sample)
    
    if sample["multi_language"] == False and sample["Language_1"] == "English":
        year = int(sample["raw_date"])
        author = sample["name"]
        all_names = sample["all_names"] # all names associated with publication
        ocr_score = sample["mean_wc_ocr"]
        ocr_score_std = sample["std_wc_ocr"]
        text = sample["text"] # sample text
        title = sample["title"] # book title
        record_id = sample["record_id"] # book id

        if ocr_score < 0.6 or ocr_score_std > 0.2:
            continue

        if author and author in seen_authors:
            continue

        if all_names in seen_all_names:
            continue

        if record_id in seen_ids:
            continue

        ready_sample = {"book_id": record_id, "year": year, "text": text}

        if (year >= 1600 and year < 1700) and (len(seventeenth_century) < 50):
            seventeenth_century.append(ready_sample)
            seen_authors.add(author)
            seen_all_names.add(all_names)
            seen_ids.add(record_id)

        elif (year >= 1700 and year < 1800) and (len(eighteenth_century) < 50):
            eighteenth_century.append(ready_sample)
            seen_authors.add(author)
            seen_all_names.add(all_names)
            seen_ids.add(record_id)
        
        elif (year >= 1800 and year < 1900) and (len(nineteenth_century) < 50):
            nineteenth_century.append(ready_sample)
            seen_authors.add(author)
            seen_all_names.add(all_names)
            seen_ids.add(record_id)

        elif len(seventeenth_century) == 50 and len(eighteenth_century) == 50 and len(nineteenth_century) == 50:
            break