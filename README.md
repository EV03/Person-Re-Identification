# Local Person Re-Identification - Masterprojekt

`main` enthält den ReID-Kern für die dokumentierten Projektversuche:
Video oder Webcam, YOLO-Personendetektion, ByteTrack/BoT-SORT, Person-Crops,
Qualitätsprüfung, OSNet/Farbhistogramm, synthetische Personen-IDs und SQLite.

**Start für Mitwirkende:** [Codebase Guide](docs/CODEBASE_GUIDE.md).
**Umfang und offene Voraussetzungen:** [Evaluationsstand](docs/EVALUATION_SCOPE.md).
**Versuchsplan und Paper:** [LaTeX-Quelle](docs/technische_systemdokumentation.tex).

Die quantitative Evaluation steht noch aus. B0/A1/A2 verwenden dokumentierte
ReID-Gewichte beziehungsweise Farbhistogramme. Vollständige Frame-Exporte,
isolierte Versuchsläufe und technische Laufmanifeste sind implementiert.
Vor der eigentlichen Messung fehlen noch annotierte Pilot-/Testclips und deren
Metrikauswertung; technische Smoke-Tests belegen keine Erkennungsgenauigkeit.

## Branches

- `main`: dokumentierter ReID-Kern und seine Tests.
- `codex/research-extensions`: gesicherter vorheriger Stand mit Fußballmodulen,
  Bewegungsdiagnostik und den bisherigen Dokumentationsfassungen.
- `Frames_Optimization` und die bisherigen Fix-Branches bleiben erhalten.

Die eigene Bewegungsdiagnostik gehört nicht zum Hauptpfad. Der verwendete Tracker
verfolgt Personen weiterhin zeitlich. Gehen, Kreuzung und Rückkehr bleiben
Versuchsszenarien; dafür sind keine eigenen Geschwindigkeitspfeile erforderlich.

## Start unter Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/ui/streamlit_app.py
```

Für OSNet zusätzlich:

```powershell
python -m pip install -r requirements-optional-reid.txt
```

Der Adapter unterstützt beide üblichen Torchreid-Paketlayouts. `gdown`,
`tensorboard` und die Tracking-Abhängigkeit `lap` sind ausdrücklich angegeben.
Die lokal technisch geprüfte Umgebung ist in `requirements-evaluation-lock.txt`
festgehalten; für andere Python-/CUDA-Umgebungen ist ein neuer Smoke-Test nötig.

OSNet nutzt explizit die öffentlichen MSMT17-ReID-Gewichte, ohne eigenes Training.
Download, Herkunft, Prüfsumme und Modelltest: [MODEL_WEIGHTS.md](docs/MODEL_WEIGHTS.md).
Gewichte sind nicht in Git. Fehlende oder inkompatible Gewichte brechen den Lauf
ab, statt unbemerkt ImageNet- oder Zufallsgewichte zu verwenden.
Die Ultralytics-Konfiguration liegt standardmäßig in `data/cache/ultralytics`;
automatische Paketinstallationen während der Verarbeitung sind deaktiviert.

## Versuchskonfigurationen

| CLI-Preset | Paper | Encoder | Kandidaten-/Updateschwelle |
|---|---|---|---|
| `default` | B0 | OSNet | 0,55 / 0,65 |
| `colorhist` | A1 | HSV-Farbhistogramm | 0,55 / 0,65 |
| `no_quality_thresholds` | A2 | OSNet | 0 / 0 |

Alle Presets verwenden zunächst YOLOv8n, ByteTrack, Cosine-Schwellwert 0,82,
Detektionskonfidenz 0,35, Eingangsgröße 640, drei Initialbeobachtungen und Updates
alle fünf Frames. Die Mindest-Crop-Größe beträgt 30 x 80 Pixel.
A2 behält Mindestgrößen, Qualitätsgewichtung und Snapshot-Auswahl bei.

Für einen vollständigen Clip ausdrücklich `--max-frames 0` verwenden; der
interaktive Standard begrenzt den Lauf auf 500 Frames.

```powershell
python -m app.main --source data/input/pilot.mp4 --mode colorhist --max-frames 0
python -m app.main --source data/input/pilot.mp4 --mode default --max-frames 0
python -m app.main --source data/input/pilot.mp4 --mode no_quality_thresholds --max-frames 0
```

Diese Befehle verwenden standardmäßig dieselbe Datenbank. Unabhängige
Versuchseinheiten brauchen getrennte Ausgangszustände und Pfade; die Befehle
allein stellen noch keinen isolierten Vergleichslauf her.

Für die eigentliche Evaluation stattdessen den isolierten Versuchsstarter nutzen:

```powershell
python -m app.evaluation --sources data/input/test.mp4 --repetitions 3
```

Das verarbeitet das vollständige Video mit B0/A1/A2, je Variante/Wiederholung mit
einer neuen Datenbank. Zusammengehörige Registrierung/Rückkehr (UC-12):

```powershell
python -m app.evaluation --sources data/input/registrierung.mp4 data/input/rueckkehr.mp4 --modes default --repetitions 3
```

Alle übergebenen Quellen gehören **einer** Versuchseinheit an und teilen deren
Profile; pro Quelle wird trotzdem ein frischer Tracker verwendet. Unabhängige
Szenarien separat starten. Anleitung und Exportformat: [EVALUATION_RUNBOOK.md](docs/EVALUATION_RUNBOOK.md).
In der UI ist "Isolierter Lauf (neue Datenbank)" standardmäßig aktiv.

CLI-Optionen überschreiben das Preset. In der UI können Parameter ebenfalls
angepasst werden. Es gibt einen gemeinsamen Editor: Preset laden, Parameter
bearbeiten und entweder starten oder die aktuellen Werte als neues Preset speichern.
Alle Laufzeitparameter aus `PipelineConfig` haben genau ein Eingabefeld. Dazu
gehören alle vier konfigurierbaren Konfidenz-/Matching-/Qualitätsschwellen,
Crop-Mindestgrößen, Padding und zeitliche Parameter. Schwellen lassen sich auch
als genaue Dezimalwerte eingeben. Interne Tracker-Schwellen gehören zur gewählten
Tracker-YAML; die festen Konstanten der Qualitätsheuristik werden nicht verändert.
Die UI erlaubt Detektionskonfidenz ab 0,0001, nicht exakt null: Ultralytics würde
den Nullwert beim Tracking durch seinen Standardwert 0,1 ersetzen.

Änderungen überleben normale UI-Neuausführungen. Ein Presetwechsel lädt dessen
gespeicherte Werte; "Änderungen verwerfen / Preset neu laden" setzt den Editor zurück.
Ein veränderter Lauf wird als "(geändert)" markiert. Die effektive Konfiguration
ist vor dem Start einsehbar und wird vollständig in den Laufmetadaten gespeichert.
Nach dem Speichern wird das neue Preset automatisch geladen. Referenzpresets und
bestehende eigene Presets können dabei nicht überschrieben werden.

Quelle, Vorschauanzeige und Vorschau-Breite sind keine Pipeline-Presetparameter.
Qualitätsschwellen auf null deaktivieren nur ihre jeweiligen Gates; eine
Crop-Mindestgröße von null deaktiviert diese Größengrenze. Updates müssen weiterhin
beide Qualitätsschwellen erfüllen. Für die finale Evaluation Einstellungen auf
Pilotdaten festlegen und vor den Testclips einfrieren.

Eigene ReID-Presets liegen in `data/modes/reid_presets.json`.
Die bisherige `custom_modes.json` bleibt unverändert und wird hier nicht geladen.

## Verarbeitung und Daten

```text
Video -> YOLO/Tracking -> unveränderter Person-Crop -> Qualitätsprüfung
      -> Initialpuffer / Encoder -> Profilabgleich -> SQLite
      -> separate Bildkopie mit Annotation -> MP4 / Vorschau
```

`track_id` ist eine Kennung des Trackers; `person_id` bezeichnet ein persistentes
synthetisches Profil. Bei neuen Tracks werden mehrere gute Beobachtungen gesammelt.
Bei bekannten Tracks werden qualitätsgefilterte Profilupdates vorgenommen.
Ein erneuter globaler ReID-Abgleich bekannter Tracks findet aktuell nicht statt.

SQLite enthält `persons`, `events` und `analysis_runs`. Bestehende Datenbanken
werden nicht bereinigt: Zusätzliche historische Tabellen und alte Payloads bleiben
erhalten, werden vom ReID-Kern aber nicht verwendet. Ereignisse entstehen nur bei
ausgewählten Matches und Updates. Zusätzlich erhält jeder Lauf einen vollständigen
`frames.jsonl`-Export, eine MOT-Ansicht echter Track-IDs und ein `manifest.json`.
Entscheidungsframe und Snapshotframe werden getrennt gespeichert; Personen-IDs
werden in früheren Frames nicht rückwirkend ergänzt. Ohne Tracker-ID bleibt eine
Box ungetrackt und geht nicht in die Profilbildung ein.

Standardpfade:

- `data/input/`: Eingabevideos
- `data/db/reid.sqlite3`: Profile, Ereignisse und Läufe
- `data/snapshots/`: Personenausschnitte
- `data/output/`: annotierte Videos
- `data/output/runs/<run_id>/`: Exporte/Manifeste interaktiver geteilter Läufe
- `data/snapshots/<run_id>/`: kollisionsfreie Personenausschnitte
- `data/experiments/<unit_id>/`: isolierte Datenbank, Snapshots, Ausgaben und Manifeste

Pfade lassen sich über die Variablen aus `.env.example` einstellen. Synthetische
IDs machen erkennbare Aufnahmen nicht anonym. Eingaben, Datenbanken und generierte
Ausgaben gehören nicht in Git.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Die Regressionstests laufen ohne Kamera und ohne Modell-Downloads. Sie prüfen
Ressourcenfreigabe, Uploads, Pfadschutz, vollständiges UI-Editieren/Speichern/Laden,
die tatsächlich gestartete Konfiguration, Datenhaltung und die Trennung von
Analysebildern und Annotation. Ein Modell-Smoke-Test und die quantitative
Evaluation sind davon getrennte Prüfungen.

Echte Modell-/Video-Smoke-Tests nach Einrichtung der Gewichte:

```powershell
$env:REID_REAL_SMOKE = '1'
python -m unittest tests.test_real_pipeline_smoke -v
Remove-Item Env:REID_REAL_SMOKE
```

Die erzeugten Testvideos enthalten keine Personen. Für Identitätsentscheidungen
prüfen zusätzliche Regressionstests definierte Box-/Embedding-Sequenzen.

## Reset

```powershell
python scripts/reset_db.py
```

Das löscht die konfigurierte Datenbank einschließlich WAL/SHM, Snapshots und
Ausgabevideos. Nur Ziele unterhalb von `data/` werden akzeptiert; `data/` selbst
und Ziele außerhalb werden abgewiesen. Vor einem Reset benötigte Versuchsdaten
sichern. Für Experimentisolation eigene Pfade verwenden.
