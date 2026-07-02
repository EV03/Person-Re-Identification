# Single-Person Evaluation für ReID-Tests

Diese Erweiterung ist für kontrollierte Tests gedacht, bei denen pro Video genau eine echte Zielperson sichtbar ist. Sie passt zu den Testordnern wie:

```text
Default/
GelbesTshirt/
MitBrille/
GelbesTshirtMitBrille/
```

Gemessen werden nicht nur die entstandenen Personen, sondern zusätzlich:

- Tracking-Stabilität innerhalb eines Videos,
- ReID-Wiedererkennung über Videos,
- Fragmentierung durch mehrere `person_id`s,
- Robustheit bei Bedingungen wie Brille, gelbes Shirt oder Kombinationen.

## Enthaltene Änderungen

```text
app/pipeline/orchestrator.py
app/evaluation/__init__.py
app/evaluation/db_snapshot.py
app/evaluation/event_logger.py
app/evaluation/manifest.py
app/evaluation/metrics.py
scripts/evaluate_single_person_videos.py
docs/EVALUATION_SINGLE_PERSON_USAGE.md
```

## 1. Manifest aus Testdaten erzeugen

Aus dem Projektroot ausführen:

```powershell
python scripts/evaluate_single_person_videos.py init-manifest `
  --video-root "E:\Person-Re-Identification\Test-daten" `
  --output "data\evaluation_manifest.csv"
```

Danach die CSV öffnen. Sie sieht ungefähr so aus:

```csv
phase,video_path,condition,expected_person_id,notes
test,E:\Person-Re-Identification\Test-daten\Default\Tim_vorne.mp4,Default,,
test,E:\Person-Re-Identification\Test-daten\GelbesTshirt\Tim_vorne.mp4,GelbesTshirt,,
```

Für das Kalibrier-Video die Spalte `phase` auf `calibration` setzen:

```csv
phase,video_path,condition,expected_person_id,notes
calibration,E:\Person-Re-Identification\Test-daten\Default\Tim_Allaround.mp4,Default,,Allround-Kalibrierung weißes Shirt
test,E:\Person-Re-Identification\Test-daten\Default\Tim_vorne.mp4,Default,,Baseline-Test
test,E:\Person-Re-Identification\Test-daten\MitBrille\Tim_links.mp4,MitBrille,,Brille + weißes Shirt
test,E:\Person-Re-Identification\Test-daten\GelbesTshirt\Tim_vorne.mp4,GelbesTshirt,,Gelbes Shirt
test,E:\Person-Re-Identification\Test-daten\GelbesTshirtMitBrille\Tim_vorne.mp4,GelbesTshirtMitBrille,,Gelbes Shirt + Brille
```

Wenn `expected_person_id` leer ist, wird nach der Kalibrierung automatisch die dominante erkannte Person-ID verwendet, z. B. `person_000001`.

## 2. Evaluation mit fixer Datenbank starten

```powershell
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode fixed_db `
  --max-frames 0
```

`--max-frames 0` bedeutet: vollständiges Video verarbeiten.

Der Ablauf ist:

```text
Kalibrier-Video -> kalibrierte SQLite-DB speichern
Testvideo 1 -> Kopie der kalibrierten DB verwenden -> Werte speichern
Testvideo 2 -> wieder neue Kopie der kalibrierten DB verwenden -> Werte speichern
...
```

Dadurch sind die Testvideos vergleichbar, weil jedes Video mit demselben DB-Stand startet.

## 3. Praxislauf mit wachsender Datenbank

Zusätzlich kann ein realistischerer Lauf getestet werden:

```powershell
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode learn_through `
  --max-frames 0
```

Dabei wird nach der Kalibrierung eine DB verwendet, die über alle Testvideos hinweg weiterwächst.

Das ist gut für einen Praxislauf, aber schlechter für saubere Einzelmetriken.

## 4. Ergebnisstruktur

Pro Lauf wird ein Ordner erstellt:

```text
data/evaluation_runs/eval_YYYYMMDD_HHMMSS/
├── run_config.json
├── calibration_state.json
├── all_videos_summary.csv
├── calibration/
│   └── calibrated_reid.sqlite3
├── video_databases/
│   ├── Tim_vorne.sqlite3
│   └── ...
└── videos/
    └── Tim_vorne_Default_test/
        ├── events.csv
        ├── events.jsonl
        ├── summary.json
        └── summary.md
```

## 5. Wichtigste Werte

### Tracking-Stabilität

```text
unique_track_ids
dominant_track_ratio
track_switch_count
tracking_rating
```

Interpretation:

```text
1 Track-ID und hohe dominant_track_ratio = stabil
mehrere Track-IDs oder viele Wechsel = Tracking instabil
```

### ReID-Wiedererkennung

```text
unique_person_ids
dominant_person_id
dominant_person_ratio
expected_person_ratio
new_person_count_from_db
reid_rating
fragmentation_rating
```

Interpretation:

```text
unique_person_ids = 1 = gut
unique_person_ids > 1 = gleiche echte Person wurde fragmentiert
expected_person_ratio hoch = erwartete Person wurde wiedererkannt
```

### Outfit-Robustheit

Outfit-Robustheit ergibt sich aus dem Vergleich der `expected_person_ratio` pro Bedingung:

```text
Default:              95 %
MitBrille:            90 %
GelbesTshirt:         70 %
GelbesTshirtBrille:   55 %
```

Dann wäre die Interpretation:

```text
Brille stört wenig.
Gelbes Shirt stört stärker.
Gelbes Shirt + Brille stört am stärksten.
```

## 6. SQLite-Sicherung

Bei SQLite reicht meistens die `.sqlite3`-Datei. Falls daneben Dateien liegen, gehören diese ebenfalls zum aktiven DB-Zustand:

```text
reid.sqlite3
reid.sqlite3-wal
reid.sqlite3-shm
```

Das Evaluationsskript nutzt für Snapshots eine SQLite-Backup-Funktion und erzeugt daraus eine konsistente Datei wie:

```text
calibrated_reid.sqlite3
```

## 7. Bewertung nur über Personenzahl reicht nicht

Nur zu zählen, wie viele Personen entstehen, ist hilfreich, aber nicht vollständig.

Beispiel:

```text
Soll: 1 echte Person
Ergebnis: person_000001, person_000002, person_000003
```

Das ist Fragmentierung. Zusätzlich muss aber geprüft werden:

```text
Welche ID war dominant?
Wie oft wurde die erwartete ID getroffen?
Gab es Track-ID-Wechsel?
Passiert das nur bei Brille/Gelb oder schon im Default?
```

Deshalb erzeugt das Skript `events.csv`, `summary.json`, `summary.md` und `all_videos_summary.csv`.

---

## Neuer Modus: adaptive_calibration

Der Modus `adaptive_calibration` bildet den gewünschten Praxisfall ab:

```text
Allaround-Video = Startprofil
weitere Testvideos = Profil darf wachsen
```

Start:

```powershell
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode adaptive_calibration `
  --max-frames 0
```

Das Allaround-Video muss im Manifest als `calibration` markiert sein. Alle anderen Videos bleiben `test`.

Ausgegeben werden zusätzlich:

- `primary_person_id`
- `primary_person_ratio`
- `adaptive_reid_rating`
- `profile_growth_event_count`
- `profile_fragmentation_index`

Die genaue Beschreibung steht in:

```text
docs/EVALUATION_ADAPTIVE_CALIBRATION_USAGE.md
```
