"""Fetch licensed iNaturalist observations and photos for project 41347.

Outputs
-------
data/raw/observations.parquet
    One row per observation.
data/raw/photos.parquet
    One row per photo, linked by observation_id.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ID = 41347
QUALITY_GRADE = "research"
PER_PAGE = 200

OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
TAXA_URL = "https://api.inaturalist.org/v1/taxa"

OUTPUT_DIR = Path("data/raw")
OBSERVATIONS_PATH = OUTPUT_DIR / "observations.parquet"
PHOTOS_PATH = OUTPUT_DIR / "photos.parquet"

CC_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-nc",
    "cc-by-sa",
    "cc-by-nc-sa",
    "cc-by-nd",
    "cc-by-nc-nd",
}

TAXONOMIC_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")

OBSERVATION_PARAMS = {
    "project_id": PROJECT_ID,
    "quality_grade": QUALITY_GRADE,
    "per_page": PER_PAGE,
    "licensed": "true",
    "photo_licensed": "true",
    "fields": ",".join(
        [
            "id",
            "uri",
            "quality_grade",
            "license_code",
            "taxon",
            "photos",
            "ofvs",
            "geojson",
            "positional_accuracy",
            "place_country_name",
            "place_state_name",
            "place_county_name",
            "place_town_name",
        ]
    ),
}


def build_session() -> requests.Session:
    """Create an HTTP session that retries temporary API failures."""
    retry = Retry(
        total=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "ecological-informatics-dataset-scraper/1.0",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    request_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    """Request one API page and return its JSON response."""
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    time.sleep(request_delay_seconds)
    return data


def fetch_all_observations(session: requests.Session) -> list[dict[str, Any]]:
    """Fetch all matching observations without crossing iNaturalist's 10k page window."""
    observations: list[dict[str, Any]] = []
    id_above: int | None = None
    batch_number = 1

    while True:
        params = {
            **OBSERVATION_PARAMS,
            "page": 1,
            "order_by": "id",
            "order": "asc",
        }

        if id_above is not None:
            params["id_above"] = id_above

        data = get_json(session, OBSERVATIONS_URL, params)
        batch = data.get("results", []) or []

        if not batch:
            break

        observations.extend(batch)
        id_above = int(batch[-1]["id"])

        print(
            f"Fetched observation batch {batch_number}: "
            f"{len(batch)} records ({len(observations):,} total)"
        )

        if len(batch) < PER_PAGE:
            break

        batch_number += 1

    return observations


def taxon_path_ids(taxon: dict[str, Any]) -> list[int]:
    """Return ancestor taxon IDs plus the observation taxon's own ID."""
    ids = [
        int(taxon_id)
        for taxon_id in (taxon.get("ancestor_ids") or [])
        if taxon_id is not None
    ]

    if taxon.get("id") is not None:
        ids.append(int(taxon["id"]))

    return list(dict.fromkeys(ids))


def fetch_taxa_lookup(
    session: requests.Session,
    taxon_ids: set[int],
    chunk_size: int = 200,
) -> dict[int, dict[str, Any]]:
    """Fetch taxon records needed to reconstruct kingdom-to-species lineage."""
    lookup: dict[int, dict[str, Any]] = {}
    ordered_ids = sorted(taxon_ids)

    for start in range(0, len(ordered_ids), chunk_size):
        chunk = ordered_ids[start : start + chunk_size]
        data = get_json(
            session,
            TAXA_URL,
            {
                "id": ",".join(map(str, chunk)),
                "per_page": len(chunk),
            },
        )

        for taxon in data.get("results", []):
            if taxon.get("id") is not None:
                lookup[int(taxon["id"])] = taxon

    return lookup


def lineage_by_rank(
    taxon: dict[str, Any],
    taxa_lookup: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map canonical taxonomic ranks to taxon ID and scientific name."""
    lineage: dict[str, dict[str, Any]] = {}

    for taxon_id in taxon_path_ids(taxon):
        taxon_record = taxa_lookup.get(taxon_id)
        if not taxon_record:
            continue

        rank = taxon_record.get("rank")
        name = taxon_record.get("name")

        if rank in TAXONOMIC_RANKS and name:
            lineage[rank] = {"id": taxon_id, "name": name}

    return lineage


def get_predator_prey_role(ofvs: list[dict[str, Any]] | None) -> Any:
    """Return the value of the 'ID meant for organism being eaten' observation field."""
    for ofv in ofvs or []:
        field_name = (ofv.get("name") or "").lower()
        if "id meant for" in field_name and "organism being eaten" in field_name:
            return ofv.get("value")
    return None


from urllib.parse import urlparse


def get_partner_observation_url(
    ofvs: list[dict[str, Any]] | None,
) -> str | None:
    """Return the URL stored in the 'url for "partner" observation' field."""
    for ofv in ofvs or []:
        field_name = (ofv.get("name") or "").strip().lower()

        if 'url for "partner" observation' in field_name:
            return ofv.get("value")

    return None


def get_partner_observation_id(
    ofvs: list[dict[str, Any]] | None,
) -> int | None:
    """Extract the numeric partner observation ID from its iNaturalist URL."""
    partner_url = get_partner_observation_url(ofvs)

    if not partner_url:
        return None

    # Handles values such as:
    # https://www.inaturalist.org/observations/123456789
    try:
        path_parts = urlparse(str(partner_url)).path.rstrip("/").split("/")
        return int(path_parts[-1])
    except (ValueError, TypeError, IndexError):
        return None


def build_tables(
    observations: list[dict[str, Any]],
    taxa_lookup: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observation and photo tables using CC-licensed records only."""
    observation_rows: list[dict[str, Any]] = []
    photo_rows: list[dict[str, Any]] = []

    for observation in observations:
        observation_license = observation.get("license_code")
        if observation_license not in CC_LICENSES:
            continue

        observation_id = observation.get("id")
        taxon = observation.get("taxon") or {}
        lineage = lineage_by_rank(taxon, taxa_lookup)

        coordinates = (observation.get("geojson") or {}).get("coordinates") or [
            None,
            None,
        ]
        longitude = coordinates[0] if len(coordinates) > 0 else None
        latitude = coordinates[1] if len(coordinates) > 1 else None

        valid_photos = [
            photo
            for photo in (observation.get("photos") or [])
            if photo.get("license_code") in CC_LICENSES
        ]

        row = {
            "observation_id": observation_id,
            "observation_uri": observation.get("uri"),
            "quality_grade": observation.get("quality_grade"),
            "observation_license_code": observation_license,
            "predator_prey_role": get_predator_prey_role(observation.get("ofvs")),
            "taxon_id": taxon.get("id"),
            "taxon_name": taxon.get("name"),
            "taxon_rank": taxon.get("rank"),
            "preferred_common_name": taxon.get("preferred_common_name"),
            "latitude": latitude,
            "longitude": longitude,
            "positional_accuracy": observation.get("positional_accuracy"),
            "country": observation.get("place_country_name"),
            "state": observation.get("place_state_name"),
            "county": observation.get("place_county_name"),
            "town": observation.get("place_town_name"),
            "photo_count": len(valid_photos),
        }

        for rank in TAXONOMIC_RANKS:
            row[f"{rank}_id"] = lineage.get(rank, {}).get("id")
            row[f"{rank}_name"] = lineage.get(rank, {}).get("name")

        observation_rows.append(row)

        for photo in valid_photos:
            photo_rows.append(
                {
                    "observation_id": observation_id,
                    "photo_id": photo.get("id"),
                    "photo_license_code": photo.get("license_code"),
                    "url": photo.get("url"),
                    "square_url": photo.get("square_url"),
                    "small_url": photo.get("small_url"),
                    "medium_url": photo.get("medium_url"),
                    "large_url": photo.get("large_url"),
                    "original_url": photo.get("original_url"),
                }
            )

    observations_df = pd.DataFrame(observation_rows).drop_duplicates(
        subset=["observation_id"]
    )
    photos_df = pd.DataFrame(photo_rows).drop_duplicates(subset=["photo_id"])

    return observations_df, photos_df


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with build_session() as session:
    observations = fetch_all_observations(session)

    all_taxon_ids: set[int] = set()
    for observation in observations:
        all_taxon_ids.update(taxon_path_ids(observation.get("taxon") or {}))

    taxa_lookup = fetch_taxa_lookup(session, all_taxon_ids)

observations_df, photos_df = build_tables(observations, taxa_lookup)

observations_df.to_parquet(OBSERVATIONS_PATH, index=False)
photos_df.to_parquet(PHOTOS_PATH, index=False)

print(f"Saved {len(observations_df):,} observations -> {OBSERVATIONS_PATH}")
print(f"Saved {len(photos_df):,} photos -> {PHOTOS_PATH}")
