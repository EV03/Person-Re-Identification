# Local Person Re-Identification

Lokale Pipeline für videobasierte Personen-Wiedererkennung. Das Projekt erkennt
Personen mit YOLO, stabilisiert Boxen mit ByteTrack oder BoT-SORT, erzeugt
OSNet-Embeddings und verwaltet synthetische Personenprofile in SQLite.

## Ziel und Abgrenzung

Ziel ist ein reproduzierbarer Versuchsaufbau für diese Fragen:

- Erkennt das System eine bereits registrierte Person nach einer Rückkehr?
- Wie wirken Crop-Qualitätsfilter und Ähnlichkeitsschutz auf Profilupdates?
- Welche Entscheidungen entstehen pro Frame und mit welcher Konfiguration?

Das System ist kein biometrisches Produkt und garantiert keine reale Identität.
`person_id` bezeichnet nur ein lokales synthetisches Profil. Die Aufnahmen bleiben
personenbezogene Daten und müssen entsprechend geschützt werden.

## Installation und Start

Voraussetzung: Python 3.10 oder neuer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional-reid.txt
python -m streamlit run app/ui/streamlit_app.py
```

Für die exakt geprüfte Evaluationsumgebung kann stattdessen
`requirements-evaluation-lock.txt` verwendet werden. OSNet benötigt den
dokumentierten Checkpoint. Download, Prüfsumme und Modell-Smoke-Test stehen in
[docs/MODEL_WEIGHTS.md](docs/MODEL_WEIGHTS.md).

Ein einzelner vollständiger CLI-Lauf:

```powershell
python -m app.main --source data/input/pilot.mp4 --mode default --max-frames 0
```

Ohne `--max-frames 0` verarbeitet der interaktive Standard höchstens 500 Frames.

## Pipeline und Methodik

```text
Video/Webcam
  -> YOLO-Personendetektion und ByteTrack/BoT-SORT
  -> Überlappungsschutz und unveränderter Person-Crop
  -> Qualitätsprüfung
  -> OSNet-Embedding
  -> Initialpuffer oder geschütztes Profilupdate
  -> Cosine Matching gegen SQLite-Profile
  -> Frameprotokoll, Snapshots, Manifest und annotiertes MP4
```

`app/pipeline/orchestrator.py` besitzt Reihenfolge und lauflokalen Zustand. Die
großen Verarbeitungsschritte sind getrennt: Frame-Verarbeitung, Überlappung,
Qualitätsentscheidung, Encoding, erste Identitätsentscheidung und Profilupdate.
Diese Trennung ändert keine öffentliche Schnittstelle und macht jede Regel einzeln
nachvollziehbar.

### Tracking und Identitäten

Der Tracker vergibt kurzlebige `track_id`-Werte. Die ReID-Schicht ordnet ihnen
persistente `person_id`-Profile zu. Sichtbare bekannte IDs werden vor neuen
Suchvorgängen reserviert, damit zwei Personen im selben Frame nicht dasselbe
Profil erhalten. Boxen ohne Tracker-ID werden protokolliert, aber nicht zur
Profilbildung genutzt.

### Qualitätsfilter

Geometrisch ungültige oder zu kleine Crops werden vor dem Encoder verworfen. Der
Qualitätswert kombiniert Schärfe, Größe, Helligkeit, Seitenverhältnis, Randkontakt
und Detektionskonfidenz. Neue Tracks erfüllen zusätzlich Schärfe- und
Seitenverhältnisgrenzen. Standardmäßig werden fünf akzeptierte Beobachtungen
qualitätsgewichtet kombiniert; der beste Crop wird Snapshot.

Überlappende Personen starten eine Abkühlphase. Kandidaten eines noch unbekannten
Tracks werden beim Überlappen verworfen, damit Ansichten vor und nach einem
möglichen Tracker-ID-Wechsel nicht vermischt werden.

### Matching und Profilupdates

Embeddings sind normalisierte 512-dimensionale OSNet-Vektoren. Das Matching nutzt
Cosine Similarity. Für ein neues Profil wird die gewichtete Summe aller akzeptierten
Initialbeobachtungen gespeichert. Bekannte Tracks werden im Standard alle fünf
Frames geprüft. Ein Update muss Crop-Qualität und `min_update_similarity` erfüllen;
eine Ablehnung verändert das Profil nicht.

Jede Encoder-Architektur und jeder Checkpoint-Inhalt erhält einen eigenen
Datenbank-Namensraum. Gleiche Dimensionen reichen nicht zur sicheren Mischung
verschiedener Embedding-Modelle.

## Konfiguration und Profile

Die zentralen Standardwerte stehen in `app/config.py`. Eingebaute Varianten werden
in `app/modes/default_mode.py` erzeugt. Eigene, über die UI gespeicherte Presets
liegen in `data/modes/reid_presets.json`.

| Preset | Zweck | Abweichung |
|---|---|---|
| `default` | B0-Basis | Qualitätsfilter und Update-Schutz aktiv |
| `no_quality_thresholds` | A2-Ablation | Qualitätsgrenzen auf null |
| `no_update_similarity` | A3-Ablation | Update-Schutz mit `-1` deaktiviert |

Wichtige Standardwerte: Matching 0,75, Update-Ähnlichkeit 0,75,
Detektionskonfidenz 0,35, fünf Initialbeobachtungen, Update alle fünf Frames,
Crop mindestens 30 x 80 Pixel und elf Frames Überlappungs-Abkühlzeit.

Änderungen sind an drei Stellen möglich:

- Dauerhafter Code-Standard: `app/config.py`, Klasse `PipelineSettings`.
- Eingebautes Vergleichsprofil: `app/modes/default_mode.py`.
- Eigenes Versuchspreset: UI speichern oder `data/modes/reid_presets.json` anlegen.

CLI-Werte wie `--threshold`, `--update-similarity`, `--device` und `--checkpoint`
überschreiben das gewählte Preset nur für diesen Lauf. Die effektive Konfiguration
wird vollständig im Manifest gespeichert.

## Teststaffelung der Profile

Die Reihenfolge verhindert, dass Testclips zur nachträglichen Optimierung dienen.

1. Lege Pilotvideos unter `data/input/pilot/` ab.
2. Wähle Schwellen und Zeitparameter nur mit diesen Pilotvideos.
3. Speichere die finale Basis über die UI als eigenes Preset, zum Beispiel
   `best-calibrated`. Speicherort ist `data/modes/reid_presets.json`.
4. Führe Unit-Tests und den Modell-Smoke-Test aus.
5. Friere Preset, Checkpoint und Testvideos ein. Ändere danach keine Schwelle anhand
   der Testergebnisse.
6. Lege zusammengehörige Registrierungs- und Rückkehrclips einer Versuchseinheit
   gemeinsam unter `data/input/test/<einheit>/` ab.
7. Starte jede unabhängige Einheit separat. Clips eines Aufrufs teilen Profile;
   verschiedene Aufrufe erhalten getrennte Datenbanken.

Beispiel für eine Einheit mit Registrierung und Rückkehr:

```powershell
python -m app.evaluation `
  --sources data/input/test/g2/register.mp4 data/input/test/g2/return.mp4 `
  --modes default no_quality_thresholds no_update_similarity `
  --device cpu
```

`--root <pfad>` ändert den Ausgabeordner. `--repetitions N` wiederholt jeden Modus
mit neuer isolierter Datenbank. Für eine andere Eingabestruktur müssen nur die
Pfade nach `--sources` geändert werden. Für neue Standardpfade ist `AppPaths` in
`app/config.py` zuständig.

Der spezielle Vier-Gruppen-Lauf erwartet ein ZIP. Sein veralteter lokaler
Standardpfad muss immer explizit überschrieben werden:

```powershell
python scripts/run_four_group_evaluation.py --zip data/input/four_groups.zip --dry-run
python scripts/run_four_group_evaluation.py --zip data/input/four_groups.zip --device cpu
```

`--dry-run` prüft ZIP, Dateizuordnung und Basispreset ohne Modelle. Abweichende
Dateinamen werden mit `--group-video G2=datei.mp4` zugeordnet. Das Skript erzeugt
für jede Gruppe und Variante einen leeren Profilbestand.

## Tests

Schnelle Staffelung:

```powershell
# 1. Syntax
python -m py_compile app/pipeline/orchestrator.py

# 2. Regressionstests ohne Kamera und Downloads
python -m unittest discover -s tests -v

# 3. Checkpoint laden, Dimension und Normalisierung prüfen
python -m app.evaluation.smoke --device cpu

# 4. Opt-in-Test mit echten Backends
$env:REID_REAL_SMOKE = '1'
$env:REID_TEST_DEVICE = 'cpu'  # alternativ cuda oder cuda:0
python -m unittest tests.test_real_pipeline_smoke -v
Remove-Item Env:REID_REAL_SMOKE
Remove-Item Env:REID_TEST_DEVICE
```

Die Unit-Tests prüfen unter anderem Ressourcenfreigabe, Profilisolation,
Qualitätsgrenzen, Überlappung, gewichtete Akkumulation, Exportdaten und Presets.
Der Modell-Smoke-Test belegt technische Lauffähigkeit, nicht die fachliche
Wiedererkennungsqualität. Diese muss mit annotierten Ereignissen geprüft werden.

## Ausgaben und Pfade

- `data/db/encoders/<encoder_key>/reid.sqlite3`: geteilter interaktiver Profilbestand
- `data/experiments/<unit_id>/`: isolierte Versuchseinheiten
- `data/output/runs/<run_id>/`: Manifeste und Frame-Exporte normaler Läufe
- `data/snapshots/<run_id>/`: kollisionsfreie Person-Crops
- `data/modes/reid_presets.json`: eigene Presets

SQLite enthält `persons`, `events` und `analysis_runs`. Jeder Lauf schreibt ein
`manifest.json`, `frames.jsonl`, eine MOT-Ansicht der Tracker-IDs und ein
annotiertes Video. Entscheidungsframe und Snapshotframe bleiben getrennt.

## Implementierter Stand und Grenzen

Implementiert sind austauschbare Tracker-, Encoder-, Repository-, Matching- und
Update-Verträge, isolierte Versuchsläufe, vollständige Konfigurationsmanifeste,
qualitätsgewichtete Profile und Schutz vor ungeeigneten oder überlappenden Crops.

Die bisherige Vier-Video-Auswertung ist historisch. Sie nutzte einen später
entfernten Abstand zwischen Initialkandidaten und darf nicht als Resultat des
aktuellen Codes zitiert werden. Werte und genaue Provenienz bleiben transparent in
[docs/evaluation/four_group_results.md](docs/evaluation/four_group_results.md).

Bekannte Grenzen:

- Die Wiedererkennung nutzt visuelle Erscheinungsmerkmale. Ähnliche Kleidung,
  Perspektivwechsel, Verdeckungen und kleine Crops verringern die Trennbarkeit.
- Eine einmal zugewiesene `person_id` wird während eines laufenden Tracks nicht
  erneut global geprüft. Der Update-Schutz kann eine falsche Erstzuordnung nicht korrigieren.
- Verfehlt ein neuer Track das richtige Profil, entsteht eine neue ID. Getrennte
  Profile derselben Person werden später nicht automatisch zusammengeführt.
- Das Matching verlangt nur den besten Score oberhalb der Schwelle. Ein
  Mindestabstand zum zweitbesten Profil fehlt.
- Die Initialentscheidung nutzt fünf fusionierte Crops, die Updateprüfung nur einen
  Crop. Derselbe Ähnlichkeitswert kann deshalb für Updates zu streng sein.
- Updates besitzen keine eigenen harten Schärfe- oder Seitenverhältnisgrenzen.
  Schwache Teilwerte können durch andere Bestandteile des Qualitätswerts ausgeglichen werden.
- Qualitätsheuristiken messen Bildtauglichkeit, nicht Personenidentität.
- Zwei Personen, vier Videos, sechs Rückkehrereignisse, ein Updatefenster und keine
  Wiederholungsläufe erlauben keine repräsentative Aussage zu Zuverlässigkeit,
  Fairness oder Schutzwirkung. Im untersuchten Fenster fehlten falsche Updatekandidaten.
- Synthetische IDs pseudonymisieren Videos, Crops und Embeddings nur. Sie
  anonymisieren die personenbezogenen Daten nicht.

## Weitere Ideen

Sinnvolle nächste Schritte, jeweils erst nach einem neuen reproduzierbaren Baseline-Lauf:

- Knappe Matches mit zusätzlichen hochwertigen Crops erneut prüfen.
- Einen Mindestabstand zwischen bestem und zweitbestem Profil evaluieren.
- Nach mehreren abgelehnten Updates einen erneuten globalen Profilabgleich auslösen.
- Eigene Update-Schärfegrenzen und eine getrennte Update-Schwelle testen.
- Mehrere ansichtsspezifische Profilprototypen oder ein begrenztes Profilgedächtnis
  statt eines unbegrenzt wachsenden Mittelwerts vergleichen.
- Eine nachgelagerte Profilkonsolidierung untersuchen. Sie soll ähnliche
  Mehrfachprofile nur mit strengeren Kriterien und zusätzlichen Beobachtungen
  zusammenführen oder ihre Trennung bestätigen.
- Erst nach belastbarer Profilkonsolidierung längere, videoübergreifend wachsende
  Profilbestände untersuchen. Sonst verfestigen sich fälschlich getrennte Profile.
- Die zweite ByteTrack-Zuordnungsstufe mit niedrigerer Detektionsschwelle prüfen.
- Mehr Personen, Kameras, Lichtbedingungen, unabhängige Annotationen und
  technische Wiederholungsläufe aufnehmen.
- Bei vollständigen Trajektorien zusätzlich IDF1 oder HOTA auswerten.

Vorerst nicht einbauen: allgemeine Bewegungsanalyse, Fußballlogik und zusätzliche
UI-Funktionen ohne Evaluationsfrage. Diese Funktionen vergrößern den Umfang, ohne
die aktuelle ReID-Hypothese zu prüfen.

## Reset

```powershell
python scripts/reset_db.py
```

Das Skript löscht konfigurierte Datenbank-, Snapshot- und Ausgabeziele nur unter
`data/`. Benötigte Versuchsdaten vorher sichern.
