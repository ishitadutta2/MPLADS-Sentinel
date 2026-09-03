"""
Synthetic MPLADS dataset generator.

Produces a realistic-looking (but entirely fictional) set of MPs,
constituencies, projects, contractors, financial transactions, geo-tagged
photo metadata and citizen reports — with a controlled fraction of
projects deliberately seeded with the fraud/irregularity patterns the
Sentinel engines are designed to catch (cost inflation, phantom works,
shell-contractor cartels, geo-mismatched or duplicate photos, and
document mismatches).

Run:  python data/generate_data.py
Writes CSVs into data/.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.config import (
    RANDOM_SEED, N_MPS, N_PROJECTS_PER_MP, N_CONTRACTORS, ANOMALY_RATE,
    DATA_DIR, REPORT_AS_OF_DATE,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------
# Reference lists (fictional but India-plausible)
# --------------------------------------------------------------------------
FIRST_NAMES = [
    "Ramesh", "Suresh", "Anita", "Priya", "Vikram", "Sunita", "Rajesh",
    "Kavita", "Manoj", "Deepa", "Arvind", "Neha", "Sanjay", "Pooja",
    "Ashok", "Meera", "Vijay", "Shalini", "Ravi", "Geeta", "Naresh",
    "Lakshmi", "Prakash", "Rekha", "Dinesh",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Singh", "Yadav",
    "Kumar", "Gupta", "Rao", "Mishra", "Chauhan", "Joshi", "Menon",
]
STATES = [
    "Uttar Pradesh", "Maharashtra", "Bihar", "West Bengal", "Tamil Nadu",
    "Karnataka", "Rajasthan", "Madhya Pradesh", "Gujarat", "Odisha",
]
PARTIES = ["Party A", "Party B", "Party C", "Party D", "Independent"]

DISTRICTS_BY_STATE = {
    s: [f"{s.split()[0]} District {i}" for i in range(1, 4)] for s in STATES
}

# Rough state centroids (lat, lon) for plausible geo-coordinates
STATE_CENTROIDS = {
    "Uttar Pradesh": (26.8, 80.9), "Maharashtra": (19.7, 75.7),
    "Bihar": (25.6, 85.1), "West Bengal": (22.9, 87.9),
    "Tamil Nadu": (11.1, 78.7), "Karnataka": (15.3, 75.7),
    "Rajasthan": (27.0, 74.2), "Madhya Pradesh": (23.5, 77.9),
    "Gujarat": (22.3, 71.2), "Odisha": (20.9, 85.1),
}

WORK_CATEGORIES = {
    "Road Construction": (300000, 4500000),
    "Community Hall": (800000, 6000000),
    "Borewell / Handpump": (80000, 350000),
    "Street Lighting": (150000, 900000),
    "School Building Repair": (200000, 2000000),
    "Drainage System": (250000, 3000000),
    "Drinking Water Supply": (300000, 2500000),
    "Public Toilet Complex": (150000, 1200000),
    "Sports Infrastructure": (400000, 3500000),
    "Library / Reading Room": (200000, 1500000),
}

STATUS_CHOICES = ["Sanctioned", "Work In Progress", "Payment Released", "Completed"]
STATUS_WEIGHTS = [0.15, 0.30, 0.25, 0.30]

# Per the eSAKSHI process: the District Authority (DA) sanctions works and
# designates an Implementing Agency (IA) — a distinct role from the
# vendor/contractor who physically executes the work under that IA.
IMPLEMENTING_AGENCY_TYPES = [
    "District Rural Development Agency",
    "Municipal Corporation Engineering Cell",
    "Zilla Parishad PWD Division",
    "Public Works Department (PWD)",
    "Panchayati Raj Engineering Department",
    "Cantonment Board Works Wing",
]


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_point_near(lat, lon, spread_deg=0.6):
    return (
        round(lat + np.random.uniform(-spread_deg, spread_deg), 6),
        round(lon + np.random.uniform(-spread_deg, spread_deg), 6),
    )


def _bounded_offset_date(base_date, min_days, max_days, hard_cap=None):
    """Adds a random day-offset to base_date, but never returns a date
    beyond hard_cap (defaults to REPORT_AS_OF_DATE) — prevents downstream
    event dates (payments, photo captures, citizen reports) from landing
    in the future just because the sanction_date itself is recent."""
    if hard_cap is None:
        hard_cap = pd.Timestamp(REPORT_AS_OF_DATE)
    available = (hard_cap - base_date).days
    if available <= min_days:
        # Sanctioned too recently for this event to plausibly have
        # happened yet within the min offset — place it as close to now
        # as possible instead of overshooting into the future.
        return min(base_date + pd.Timedelta(days=max(available, 0)), hard_cap)
    offset = int(np.random.uniform(min_days, min(max_days, available)))
    return base_date + pd.Timedelta(days=offset)


# --------------------------------------------------------------------------
# 1. MPs & constituencies
# --------------------------------------------------------------------------
def generate_mps():
    rows = []
    for i in range(1, N_MPS + 1):
        state = random.choice(STATES)
        rows.append({
            "mp_id": f"MP{i:03d}",
            "mp_name": rand_name(),
            "constituency": f"{state.split()[0]}-{random.randint(1, 40)}",
            "state": state,
            "party": random.choice(PARTIES),
            "term_start": "2024-06-01",
            "fund_allocated_total": 25_00_00_000,  # Rs 25 Cr standard MPLADS entitlement (2 installments/yr modelled as pool)
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Contractors — includes deliberately seeded shell-company clusters
# --------------------------------------------------------------------------
def generate_contractors():
    rows = []
    # A handful of "cartel" clusters: contractors that share directors /
    # registration addresses / bank accounts (proxy signals used by the
    # graph engine). ~15% of contractors belong to a cartel cluster.
    n_cartels = 6
    cartel_size = 4
    cartel_contractors = n_cartels * cartel_size

    shared_addresses = [f"Shared Complex {i}, Industrial Area" for i in range(n_cartels)]
    shared_bank_prefixes = [f"HDFC00{100+i}" for i in range(n_cartels)]
    shared_directors = [rand_name() for _ in range(n_cartels)]

    idx = 1
    for c in range(n_cartels):
        for _ in range(cartel_size):
            rows.append({
                "contractor_id": f"CTR{idx:04d}",
                "contractor_name": f"{random.choice(LAST_NAMES)} {random.choice(['Infra', 'Builders', 'Construction', 'Enterprises', 'Projects'])} Pvt Ltd",
                "registration_address": shared_addresses[c],
                "bank_account": f"{shared_bank_prefixes[c]}{idx:06d}",
                "director_name": shared_directors[c],
                "registration_year": random.randint(2018, 2023),
                "is_cartel_seed": True,
                "cartel_cluster": f"CLUSTER_{c}",
            })
            idx += 1

    used_non_cartel_directors = set(shared_directors)
    used_bank_prefixes = set()

    def _unique_bank_account(idx):
        # Same rationale as _unique_director_name(): guarantee the
        # 7-character bank-account prefix that the graph engine actually
        # keys on (bank_account[:-6] — i.e. "SBI" + the first 4 digits of
        # the random 6-digit number, since the trailing 4-digit contractor
        # index gets stripped off too) doesn't accidentally collide across
        # unrelated non-cartel firms.
        for _ in range(200):
            digits6 = random.randint(100000, 999999)
            prefix = f"SBI{str(digits6)[:4]}"
            if prefix not in used_bank_prefixes:
                used_bank_prefixes.add(prefix)
                return f"SBI{digits6}{idx:04d}"
        # exceedingly unlikely fallback
        return f"SBI{random.randint(100000,999999)}{idx:04d}{random.randint(0,9)}"

    def _unique_director_name():
        # Guarantees non-cartel contractors never accidentally share a
        # director purely by chance draw from a small name pool — that
        # coincidence would otherwise get mistaken by the graph engine for
        # a genuine shared-identity signal. Real systems key this off a
        # unique DIN/PAN, not a raw name string; we approximate that
        # uniqueness guarantee here since we don't model ID numbers.
        for _ in range(200):
            name = rand_name()
            if name not in used_non_cartel_directors:
                used_non_cartel_directors.add(name)
                return name
        # exceedingly unlikely fallback: disambiguate with a suffix
        name = f"{rand_name()} {random.randint(2,99)}"
        used_non_cartel_directors.add(name)
        return name

    for _ in range(N_CONTRACTORS - cartel_contractors):
        rows.append({
            "contractor_id": f"CTR{idx:04d}",
            "contractor_name": f"{random.choice(LAST_NAMES)} {random.choice(['Infra', 'Builders', 'Construction', 'Enterprises', 'Projects'])} Pvt Ltd",
            "registration_address": f"{random.randint(1,999)} Main Road, Sector {random.randint(1,30)}",
            "bank_account": _unique_bank_account(idx),
            "director_name": _unique_director_name(),
            "registration_year": random.randint(2005, 2023),
            "is_cartel_seed": False,
            "cartel_cluster": None,
        })
        idx += 1

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Projects — the core table, with anomaly flags seeded in for eval
# --------------------------------------------------------------------------
def generate_projects(mps_df, contractors_df):
    rows = []
    project_idx = 1
    contractor_ids = contractors_df["contractor_id"].tolist()
    cartel_contractor_ids = contractors_df[contractors_df.is_cartel_seed]["contractor_id"].tolist()

    for _, mp in mps_df.iterrows():
        state = mp.state
        centroid = STATE_CENTROIDS[state]
        district = random.choice(DISTRICTS_BY_STATE[state])

        for _ in range(N_PROJECTS_PER_MP):
            category = random.choice(list(WORK_CATEGORIES.keys()))
            lo, hi = WORK_CATEGORIES[category]
            base_cost = float(np.random.uniform(lo, hi))

            is_anomalous = np.random.random() < ANOMALY_RATE
            anomaly_types = []

            sanctioned_amount = base_cost
            expenditure = base_cost * np.random.uniform(0.85, 1.0)

            # pick contractor — anomalous projects biased toward cartel contractors
            if is_anomalous and np.random.random() < 0.5 and cartel_contractor_ids:
                contractor_id = random.choice(cartel_contractor_ids)
            else:
                contractor_id = random.choice(contractor_ids)

            lat, lon = rand_point_near(*centroid)
            photo_lat, photo_lon = lat, lon

            sanction_date = pd.Timestamp(REPORT_AS_OF_DATE) - pd.Timedelta(days=int(np.random.uniform(0, 800)))
            recommendation_date = sanction_date - pd.Timedelta(days=int(np.random.uniform(15, 90)))
            age_days = (pd.Timestamp(REPORT_AS_OF_DATE) - sanction_date).days

            # Status realistically correlates with how long ago a project
            # was sanctioned — a project sanctioned last month is very
            # unlikely to already be "Completed".
            if age_days < 90:
                status = random.choices(STATUS_CHOICES, weights=[0.55, 0.35, 0.10, 0.00])[0]
            elif age_days < 365:
                status = random.choices(STATUS_CHOICES, weights=[0.10, 0.35, 0.35, 0.20])[0]
            else:
                status = random.choices(STATUS_CHOICES, weights=[0.03, 0.12, 0.25, 0.60])[0]

            if is_anomalous:
                kind = np.random.choice(
                    ["cost_inflation", "phantom_work", "geo_mismatch", "split_tender", "duplicate_photo"],
                    p=[0.30, 0.20, 0.25, 0.15, 0.10],
                )
                if kind == "cost_inflation":
                    expenditure = base_cost * np.random.uniform(1.6, 2.8)
                    sanctioned_amount = expenditure * np.random.uniform(0.95, 1.0)
                    anomaly_types.append("cost_inflation")
                elif kind == "phantom_work":
                    status = "Completed"
                    expenditure = sanctioned_amount * np.random.uniform(0.9, 1.0)
                    # photo coordinates far from declared site -> nothing built there
                    photo_lat, photo_lon = rand_point_near(lat, lon, spread_deg=0.15)
                    anomaly_types.append("phantom_work")
                elif kind == "geo_mismatch":
                    # photo taken >5km from declared project site
                    photo_lat = lat + np.random.uniform(0.05, 0.12) * random.choice([-1, 1])
                    photo_lon = lon + np.random.uniform(0.05, 0.12) * random.choice([-1, 1])
                    anomaly_types.append("geo_mismatch")
                elif kind == "split_tender":
                    # cost sits just under a scrutiny threshold (e.g. 5,00,000)
                    sanctioned_amount = np.random.uniform(0.90, 0.99) * 500000
                    expenditure = sanctioned_amount * np.random.uniform(0.95, 1.0)
                    anomaly_types.append("split_tender")
                elif kind == "duplicate_photo":
                    anomaly_types.append("duplicate_photo")  # handled at photo-gen stage

            implementing_agency = random.choice(IMPLEMENTING_AGENCY_TYPES)

            # Per eSAKSHI process: only works the Implementing Agency has
            # explicitly "marked complete" on the portal show as completed
            # on the public dashboard — even when the work is functionally
            # finished, there is real-world lag in this final step.
            if status == "Completed":
                marked_complete_by_ia = np.random.random() > 0.12  # ~12% real-world lag
            else:
                marked_complete_by_ia = False

            rows.append({
                "project_id": f"PRJ{project_idx:05d}",
                "mp_id": mp.mp_id,
                "constituency": mp.constituency,
                "state": state,
                "district": district,
                "category": category,
                "description": f"{category} at {district}, Ward {random.randint(1,25)}",
                "sanctioned_amount": round(sanctioned_amount, 2),
                "expenditure": round(expenditure, 2),
                "unit_cost_basis": round(sanctioned_amount, 2),  # simplification: 1 unit per project
                "status": status,
                "recommendation_date": recommendation_date,
                "sanction_date": sanction_date,
                "implementing_agency": implementing_agency,
                "marked_complete_by_ia": marked_complete_by_ia,
                "contractor_id": contractor_id,
                "site_latitude": lat,
                "site_longitude": lon,
                "photo_latitude": photo_lat,
                "photo_longitude": photo_lon,
                "is_seeded_anomalous": is_anomalous,
                "seeded_anomaly_types": ",".join(anomaly_types) if anomaly_types else "",
            })
            project_idx += 1

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Financial transactions (installments / disbursals per project)
# --------------------------------------------------------------------------
def generate_transactions(projects_df):
    rows = []
    txn_idx = 1
    for _, p in projects_df.iterrows():
        if p.status == "Sanctioned":
            # No vendor payment request has been raised yet for a project
            # that's only just been sanctioned — matches the real eSAKSHI
            # workflow (IAs raise payment requests as execution proceeds).
            continue
        n_installments = random.choice([1, 2, 3])
        remaining = p.expenditure
        for i in range(n_installments):
            amt = remaining / (n_installments - i) if i < n_installments - 1 else remaining
            amt *= np.random.uniform(0.9, 1.0) if i < n_installments - 1 else 1.0
            remaining -= amt
            rows.append({
                "transaction_id": f"TXN{txn_idx:06d}",
                "project_id": p.project_id,
                "contractor_id": p.contractor_id,
                "amount": round(max(amt, 0), 2),
                "txn_date": _bounded_offset_date(p.sanction_date, 10, 400),
                "installment_no": i + 1,
            })
            txn_idx += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Photo metadata table (actual image files generated separately)
# --------------------------------------------------------------------------
def generate_photos(projects_df):
    rows = []
    photo_idx = 1
    # Pool of "reused" image signatures for duplicate-photo anomalies
    dup_pool = projects_df[projects_df.seeded_anomaly_types.str.contains("duplicate_photo", na=False)]
    dup_project_ids = dup_pool.project_id.tolist()
    dup_source_image = None

    for _, p in projects_df.iterrows():
        n_photos = random.choice([1, 2, 3])
        for i in range(n_photos):
            reused = p.project_id in dup_project_ids and i == 0
            rows.append({
                "photo_id": f"PHT{photo_idx:06d}",
                "project_id": p.project_id,
                "category": p.category,
                "latitude": p.photo_latitude + np.random.uniform(-0.0005, 0.0005),
                "longitude": p.photo_longitude + np.random.uniform(-0.0005, 0.0005),
                "capture_date": _bounded_offset_date(p.sanction_date, 5, 500),
                "is_seeded_duplicate": reused,
                "filename": "",  # filled in after image generation
            })
            photo_idx += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. Citizen reports (crowdsourced ground-truth signals)
# --------------------------------------------------------------------------
def generate_citizen_reports(projects_df):
    rows = []
    # Citizens are more likely to report on seeded-anomalous projects
    # (models real-world dynamic: irregularities draw complaints) but with
    # noise so it's not a trivial giveaway label.
    report_idx = 1
    for _, p in projects_df.iterrows():
        base_prob = 0.28 if p.is_seeded_anomalous else 0.05
        if np.random.random() < base_prob:
            complaint_type = random.choice([
                "Work not started despite funds released",
                "Poor quality of construction material",
                "Work incomplete but marked complete",
                "No such facility exists at stated location",
                "Overpricing compared to similar local works",
            ])
            rows.append({
                "report_id": f"CIT{report_idx:05d}",
                "project_id": p.project_id,
                "complaint_type": complaint_type,
                "report_date": _bounded_offset_date(p.sanction_date, 5, 600),
                "citizen_name_masked": f"Citizen_{random.randint(1000,9999)}",
                "status": random.choice(["Open", "Under Review", "Resolved"]),
            })
            report_idx += 1
    return pd.DataFrame(rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mps_df = generate_mps()
    contractors_df = generate_contractors()
    projects_df = generate_projects(mps_df, contractors_df)
    transactions_df = generate_transactions(projects_df)
    photos_df = generate_photos(projects_df)
    citizen_df = generate_citizen_reports(projects_df)

    mps_df.to_csv(DATA_DIR / "mps.csv", index=False)
    contractors_df.to_csv(DATA_DIR / "contractors.csv", index=False)
    projects_df.to_csv(DATA_DIR / "projects.csv", index=False)
    transactions_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    photos_df.to_csv(DATA_DIR / "photos.csv", index=False)
    citizen_df.to_csv(DATA_DIR / "citizen_reports.csv", index=False)

    print(f"MPs:              {len(mps_df)}")
    print(f"Contractors:      {len(contractors_df)} ({contractors_df.is_cartel_seed.sum()} in cartel clusters)")
    print(f"Projects:         {len(projects_df)} ({projects_df.is_seeded_anomalous.sum()} seeded anomalous)")
    print(f"Transactions:     {len(transactions_df)}")
    print(f"Photos:           {len(photos_df)} ({photos_df.is_seeded_duplicate.sum()} seeded duplicates)")
    print(f"Citizen reports:  {len(citizen_df)}")


if __name__ == "__main__":
    main()
