import os
import sys
import csv
import time
import django
from django.db import transaction

# Setup Django environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare_project.settings')
django.setup()

from apps.drug_checker.models import Drug, DrugInteraction

DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

CSV_FILES = [
    "ddinter_downloads_code_A.csv",
    "ddinter_downloads_code_B.csv",
    "ddinter_downloads_code_D.csv",
    "ddinter_downloads_code_H.csv",
    "ddinter_downloads_code_L.csv",
    "ddinter_downloads_code_P.csv",
    "ddinter_downloads_code_R.csv",
    "ddinter_downloads_code_V.csv"
]


def normalize_level(level_str: str) -> str:
    lvl = (level_str or "").strip().lower()
    if "major" in lvl:
        return "major"
    elif "minor" in lvl:
        return "minor"
    elif "contraindicated" in lvl or "severe" in lvl:
        return "contraindicated"
    else:
        return "moderate"


def ingest_all_ddinter_datasets():
    print("=========================================================")
    print("  INGESTING FULL DDINTER DATASET (8 CATEGORY CSV FILES)  ")
    print("=========================================================")

    t0 = time.time()
    total_rows_parsed = 0
    unique_drugs = set()
    interaction_batch = []
    
    seen_pairs = set()

    for csv_name in CSV_FILES:
        csv_path = os.path.join(DATASET_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"Skipping missing file: {csv_path}")
            continue

        print(f"Processing {csv_name}...")
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            file_count = 0
            for row in reader:
                drug_a = (row.get('Drug_A') or '').strip()
                drug_b = (row.get('Drug_B') or '').strip()
                level = normalize_level(row.get('Level', ''))

                if not drug_a or not drug_b or drug_a.lower() == drug_b.lower():
                    continue

                unique_drugs.add(drug_a.title())
                unique_drugs.add(drug_b.title())

                pair_1 = (drug_a.title(), drug_b.title())
                pair_2 = (drug_b.title(), drug_a.title())

                rec_text = f"DDInter clinical interaction check: {drug_a.title()} and {drug_b.title()} ({level.upper()}). Monitor patient parameters and review co-administration suitability."

                if pair_1 not in seen_pairs:
                    seen_pairs.add(pair_1)
                    interaction_batch.append(
                        DrugInteraction(
                            drug_a=drug_a.title(),
                            drug_b=drug_b.title(),
                            severity=level,
                            mechanism=f"DDInter dataset recorded interaction between {drug_a} and {drug_b}.",
                            recommendation=rec_text
                        )
                    )

                if pair_2 not in seen_pairs:
                    seen_pairs.add(pair_2)
                    interaction_batch.append(
                        DrugInteraction(
                            drug_a=drug_b.title(),
                            drug_b=drug_a.title(),
                            severity=level,
                            mechanism=f"DDInter dataset recorded interaction between {drug_b} and {drug_a}.",
                            recommendation=rec_text
                        )
                    )

                file_count += 1
                total_rows_parsed += 1

                if len(interaction_batch) >= 10000:
                    with transaction.atomic():
                        DrugInteraction.objects.bulk_create(interaction_batch, ignore_conflicts=True, batch_size=2000)
                    interaction_batch = []

        print(f"  Finished {csv_name}: {file_count} records processed.")

    if interaction_batch:
        with transaction.atomic():
            DrugInteraction.objects.bulk_create(interaction_batch, ignore_conflicts=True, batch_size=2000)

    print("\nIngesting unique drug names into Drug catalog...")
    drug_batch = [Drug(name=name, category="DDInter Formulated") for name in unique_drugs]
    with transaction.atomic():
        Drug.objects.bulk_create(drug_batch, ignore_conflicts=True, batch_size=2000)

    elapsed = time.time() - t0
    final_interaction_count = DrugInteraction.objects.count()
    final_drug_count = Drug.objects.count()

    print("\n=========================================================")
    print("                INGESTION SUMMARY REPORT                 ")
    print("=========================================================")
    print(f"Total CSV Rows Parsed      : {total_rows_parsed}")
    print(f"Total Unique Interactions  : {final_interaction_count}")
    print(f"Total Unique Drugs Catalog : {final_drug_count}")
    print(f"Time Taken                 : {elapsed:.2f} seconds")
    print("=========================================================")

if __name__ == "__main__":
    ingest_all_ddinter_datasets()
