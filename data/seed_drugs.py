import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'healthcare_project.settings')
django.setup()

from apps.drug_checker.models import Drug, DrugInteraction

NLEM_DRUGS = [
    {"name": "Warfarin", "generic_name": "Warfarin Sodium", "category": "Cardiovascular", "dosage_forms": "Tablet 1mg, 2mg, 5mg", "therapeutic_class": "Anticoagulant"},
    {"name": "Aspirin", "generic_name": "Acetylsalicylic Acid", "category": "Analgesic / Anti-inflammatory", "dosage_forms": "Tablet 75mg, 150mg, 300mg", "therapeutic_class": "Antiplatelet / NSAID"},
    {"name": "Metformin", "generic_name": "Metformin Hydrochloride", "category": "Endocrine", "dosage_forms": "Tablet 500mg, 850mg, 1000mg", "therapeutic_class": "Biguanide Antidiabetic"},
    {"name": "Amiodarone", "generic_name": "Amiodarone Hydrochloride", "category": "Cardiovascular", "dosage_forms": "Tablet 100mg, 200mg", "therapeutic_class": "Antiarrhythmic"},
    {"name": "Digoxin", "generic_name": "Digoxin", "category": "Cardiovascular", "dosage_forms": "Tablet 0.25mg", "therapeutic_class": "Cardiac Glycoside"},
    {"name": "Ciprofloxacin", "generic_name": "Ciprofloxacin", "category": "Anti-infective", "dosage_forms": "Tablet 250mg, 500mg", "therapeutic_class": "Fluoroquinolone Antibiotic"},
    {"name": "Tizanidine", "generic_name": "Tizanidine", "category": "Musculoskeletal", "dosage_forms": "Tablet 2mg, 4mg", "therapeutic_class": "Muscle Relaxant"},
    {"name": "Sildenafil", "generic_name": "Sildenafil Citrate", "category": "Urology", "dosage_forms": "Tablet 25mg, 50mg, 100mg", "therapeutic_class": "PDE5 Inhibitor"},
    {"name": "Nitroglycerin", "generic_name": "Glyceryl Trinitrate", "category": "Cardiovascular", "dosage_forms": "Sublingual Tablet 0.5mg", "therapeutic_class": "Nitrate Vasodilator"},
    {"name": "Paracetamol", "generic_name": "Acetaminophen", "category": "Analgesic / Antipyretic", "dosage_forms": "Tablet 500mg, 650mg", "therapeutic_class": "Antipyretic Analgesic"},
    {"name": "Ibuprofen", "generic_name": "Ibuprofen", "category": "NSAID", "dosage_forms": "Tablet 200mg, 400mg", "therapeutic_class": "NSAID"},
    {"name": "Amoxicillin", "generic_name": "Amoxicillin Trihydrate", "category": "Anti-infective", "dosage_forms": "Capsule 250mg, 500mg", "therapeutic_class": "Penicillin Antibiotic"},
    {"name": "Clopidogrel", "generic_name": "Clopidogrel Bisulfate", "category": "Cardiovascular", "dosage_forms": "Tablet 75mg", "therapeutic_class": "Antiplatelet Agent"},
    {"name": "Omeprazole", "generic_name": "Omeprazole", "category": "Gastrointestinal", "dosage_forms": "Capsule 20mg", "therapeutic_class": "Proton Pump Inhibitor"},
]

INTERACTIONS = [
    {
        "drug_a": "Warfarin",
        "drug_b": "Aspirin",
        "severity": "major",
        "mechanism": "Concomitant use of antiplatelet agents (aspirin) and anticoagulants (warfarin) synergistically increases systemic bleeding risk.",
        "recommendation": "Avoid combination unless strictly indicated for specific cardiac conditions (e.g. prosthetic mechanical valve). Monitor INR and signs of hemorrhage closely."
    },
    {
        "drug_a": "Warfarin",
        "drug_b": "Ibuprofen",
        "severity": "major",
        "mechanism": "NSAIDs inhibit platelet function and can cause GI mucosal damage, increasing Warfarin bleeding risk.",
        "recommendation": "Contraindicated for routine analgesia. Use Paracetamol as preferred pain relief in patients on Warfarin."
    },
    {
        "drug_a": "Sildenafil",
        "drug_b": "Nitroglycerin",
        "severity": "contraindicated",
        "mechanism": "Nitrates increase cGMP; PDE5 inhibition prevents cGMP degradation leading to profound, severe life-threatening hypotension.",
        "recommendation": "STRICTLY CONTRAINDICATED. Do not administer Nitroglycerin within 24 hours of Sildenafil intake."
    },
    {
        "drug_a": "Amiodarone",
        "drug_b": "Digoxin",
        "severity": "major",
        "mechanism": "Amiodarone inhibits P-glycoprotein efflux transporter, increasing serum Digoxin levels by 70-100%, leading to digoxin toxicity.",
        "recommendation": "Reduce Digoxin dose by 50% when initiating Amiodarone. Monitor serum digoxin concentrations and ECG."
    },
    {
        "drug_a": "Ciprofloxacin",
        "drug_b": "Tizanidine",
        "severity": "contraindicated",
        "mechanism": "Ciprofloxacin is a potent CYP1A2 inhibitor, raising Tizanidine plasma concentration up to 10-fold, causing severe hypotension and sedation.",
        "recommendation": "STRICTLY CONTRAINDICATED. Select alternative antibiotic (e.g., Azithromycin or Ceftriaxone)."
    },
    {
        "drug_a": "Clopidogrel",
        "drug_b": "Omeprazole",
        "severity": "moderate",
        "mechanism": "Omeprazole inhibits CYP2C19, decreasing conversion of Clopidogrel to its active metabolite, reducing antiplatelet efficacy.",
        "recommendation": "Consider non-CYP2C19 inhibiting PPI like Pantoprazole or H2 blocker like Famotidine."
    },
]

def seed_database():
    print("Seeding NLEM 2022 Drugs...")
    for d in NLEM_DRUGS:
        Drug.objects.update_or_create(
            name=d["name"],
            defaults=d
        )
    print(f"Successfully seeded {len(NLEM_DRUGS)} drugs.")

    print("Seeding DDInter Drug Interactions...")
    count = 0
    for inter in INTERACTIONS:
        DrugInteraction.objects.update_or_create(
            drug_a=inter["drug_a"],
            drug_b=inter["drug_b"],
            defaults={
                "severity": inter["severity"],
                "mechanism": inter["mechanism"],
                "recommendation": inter["recommendation"],
            }
        )
        if inter["drug_a"] != inter["drug_b"]:
            DrugInteraction.objects.update_or_create(
                drug_a=inter["drug_b"],
                drug_b=inter["drug_a"],
                defaults={
                    "severity": inter["severity"],
                    "mechanism": inter["mechanism"],
                    "recommendation": inter["recommendation"],
                }
            )
        count += 1
    print(f"Successfully seeded {count} interaction pairs.")

if __name__ == "__main__":
    seed_database()
