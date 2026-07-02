# Adaptive Kalibrierung – Single-Person-Evaluation

## Ziel

Dieser Modus bildet den gewünschten Praxisfall ab:

```text
Kalibrierungsvideo = Startprofil
weitere Videos = Profil darf kontrolliert wachsen
```

Die Kalibrierung wird dabei nicht als starre `Single Source of Truth` behandelt, sondern als initialer Startpunkt für `person_000001`. Danach darf die Datenbank aus weiteren hochwertigen Crops lernen.

## Manifest vorbereiten

In `data/evaluation_manifest.csv` muss genau das Allaround-Video als Kalibrierung markiert werden:

```csv
phase,video_path,condition,expected_person_id,notes
calibration,E:\Person-Re-Identification\Test-daten\Default\Tim_Allaround_Ohne_Details_Weises_Tshirt_Schwarze_hose.mp4,Default,,Startprofil aus Allaround-Video
test,E:\Person-Re-Identification\Test-daten\Default\Tim_Herumlaufend_Ohne_Details_Weises_Tshirt_Schwarze_hose_Wiedererkennung.mp4,Default,,
test,E:\Person-Re-Identification\Test-daten\GelbesTshirt\Tim_hinten_nach_vorne_Ohne_Details_Gelbes_Tshirt_Schwarze_hose.mp4,GelbesTshirt,,
```

Wichtig: Alle anderen Videos bleiben `test`.

## Adaptive Evaluation starten

```powershell
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode adaptive_calibration `
  --max-frames 0
```

Aliasnamen funktionieren ebenfalls:

```powershell
--test-mode adaptive
--test-mode calibration_then_adaptive
--test-mode growing_calibration
--test-mode calibration_then_grow
```

## Unterschied zu den anderen Modi

| Modus | Bedeutung | Verwendung |
|---|---|---|
| `fixed_db` | Jedes Testvideo startet mit derselben kalibrierten DB-Kopie | strenger Referenztest |
| `learn_through` | Videos laufen nacheinander, DB wächst | Praxislauf ohne explizite Kalibrierlogik |
| `adaptive_calibration` | Kalibrierung erzeugt Startprofil, danach wächst die DB über die Testvideos | empfohlener Use Case |

## Neue Ergebniswerte

Zusätzlich zu den bisherigen Kennzahlen werden diese Werte ausgegeben:

| Wert | Bedeutung |
|---|---|
| `primary_person_id` | erwartete Kalibrier-ID, sonst dominante Person-ID |
| `primary_person_ratio` | Anteil der Events, die auf das Hauptprofil fallen |
| `adaptive_reid_rating` | Bewertung nach Hauptprofil-Stabilität |
| `profile_growth_event_count` | wie oft das Hauptprofil weitergeführt/aktualisiert wurde |
| `profile_fragmentation_index` | wie viele zusätzliche Person-IDs neben der einen erwarteten Person entstanden sind |

## Ergebnisordner

Nach dem Lauf liegt alles unter:

```text
data/evaluation_runs/eval_YYYYMMDD_HHMMSS/
├── run_config.json
├── calibration_state.json
├── all_videos_summary.csv
├── calibration/
│   └── calibrated_reid.sqlite3
├── adaptive_calibration/
│   ├── reid_eval.sqlite3
│   └── final_adaptive_reid.sqlite3
└── videos/
    └── <video_name>/
        ├── events.csv
        ├── events.jsonl
        ├── summary.json
        └── summary.md
```

## Interpretation

Für den adaptiven Use Case ist nicht nur entscheidend, ob exakt `person_000001` in jedem Frame getroffen wurde. Entscheidend ist, ob die echte Person überwiegend in einem Hauptprofil zusammengeführt wird und ob die Datenbank dabei nicht unnötig viele konkurrierende IDs erzeugt.

Gute adaptive Ergebnisse sehen ungefähr so aus:

```text
primary_person_ratio hoch
profile_fragmentation_index niedrig
new_person_count_from_db niedrig
```

Schlechte adaptive Ergebnisse sehen so aus:

```text
primary_person_ratio niedrig
profile_fragmentation_index hoch
viele neue person_00000X IDs
```

## Empfohlene Vergleichsläufe

Für die wissenschaftliche Auswertung sollten drei Durchgänge erzeugt werden:

```powershell
# 1. Ohne Kalibrierung, DB wächst frei
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode learn_through `
  --max-frames 0

# 2. Starre Kalibrierung
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode fixed_db `
  --max-frames 0

# 3. Adaptive Kalibrierung
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode adaptive_calibration `
  --max-frames 0
```

Hinweis: Für Durchgang 1 sollte das Allaround-Video im Manifest als `test` markiert sein. Für Durchgang 2 und 3 sollte es als `calibration` markiert sein.
