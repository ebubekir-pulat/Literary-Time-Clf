from datasets import load_dataset
import random
import json

def write_dataset():

    DATASET = "biglam/blbooks-parquet"

    print("Dataset Source: ", DATASET, "\n")

    dataset = load_dataset(DATASET, split='train', streaming=True, trust_remote_code=True)

    rd_seed = random.seed(2)
    dataset = dataset.shuffle(seed=rd_seed, buffer_size=10_000)

    print("\nDataset Preview: ", dataset, "\n")

    # Thirds of the 19th Century
    first_third  = []    # 1800 - 1832
    second_third = []    # 1833 - 1866
    third_third  = []    # 1867 - 1899

    # Prevent repeat books and authors
    seen_authors = set()
    seen_all_names = set()
    seen_ids = set()

    dataset_full = False

    # Get 100 valid book excerpts for each century-third
    for loop in range(1000):
        if dataset_full == True:
            break

        dataset.set_epoch(loop)

        for i, sample in enumerate(dataset):
            if i == 35:
                break
            
            book_added = False
            
            if sample["multi_language"] == False and sample["Language_1"] == "English":
                try: 
                    year = int(sample["raw_date"]) 
                except: 
                    continue
                
                author = sample["name"]
                all_names = sample["all_names"]        # all names associated with publication
                ocr_score = sample["mean_wc_ocr"]
                ocr_score_std = sample["std_wc_ocr"]
                text = sample["text"]                  # sample text
                title = sample["title"]                # book title
                record_id = sample["record_id"]        # book id

                if (not ocr_score) or (not ocr_score_std) or ocr_score < 0.50 or ocr_score_std > 0.4:
                    continue

                if (not author) or author in seen_authors:
                    continue

                if (not all_names) or all_names in seen_all_names:
                    continue

                if (not record_id) or record_id in seen_ids:
                    continue

                ready_sample = {"book_id": record_id, "year": year, "text": text}

                if len(first_third) == 100 and len(second_third) == 100 and len(third_third) == 100:
                    dataset_full = True
                    break

                elif (year >= 1800 and year < 1833) and (len(first_third) < 100):
                    first_third.append(ready_sample)
                    book_added = True

                elif (year >= 1833 and year < 1867) and (len(second_third) < 100):
                    second_third.append(ready_sample)
                    book_added = True

                elif (year >= 1867 and year < 1900) and (len(third_third) < 100):
                    third_third.append(ready_sample)
                    book_added = True

                if book_added:
                    seen_authors.add(author)
                    seen_all_names.add(all_names)
                    seen_ids.add(record_id)


    print("Final Dataset Counts")
    print("1st Third: ", len(first_third))
    print("2nd Third: ", len(second_third))
    print("3rd Third: ", len(third_third), "\n")

    # Write Recorded Samples to Files
    samples = {
        "first_third": first_third,
        "second_third": second_third,
        "third_third": third_third
    }

    for name, data in samples.items():
        with open(f"../data/{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    print("Dataset Files Written")