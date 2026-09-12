# Local Person Re-Identification - Masterprojekt

`main` enthält den ReID-Kern für die dokumentierten Projektversuche:
Video oder Webcam, YOLO-Personendetektion, ByteTrack/BoT-SORT, Person-Crops,
Qualitätsprüfung, OSNet/Farbhistogramm, synthetische Personen-IDs und SQLite.

**Start für Mitwirkende:** [Codebase Guide](docs/CODEBASE_GUIDE.md).
**Umfang und offene Voraussetzungen:** [Evaluationsstand](docs/EVALUATION_SCOPE.md).
**Versuchsplan und Paper:** [LaTeX-Quelle](docs/technische_systemdokumentation.tex).

Die quantitative Evaluation steht noch aus. Die Presets bilden B0/A1/A2 ab;
Modellgewichte, vollständiger Vorhersageexport und reproduzierbare Messläufe sind
noch vor Beginn der abschließenden Evaluation zu ergänzen.

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

Die Paketstruktur von Torchreid unterscheidet sich zwischen Distributionen.
Der Adapter erwartet `torchreid.utils.FeatureExtractor`. Die Installation muss
vor den Experimenten mit einem Encoder-Test geprüft und in einer reproduzierbaren
Umgebung festgehalten werden.

Beim ersten Modellstart können Gewichte heruntergeladen werden. Aktuell übergibt
der OSNet-Adapter einen leeren Checkpoint-Pfad. Das ist keine dokumentierte Auswahl
ReID-trainierter Gewichte. Checkpoint und Herkunft müssen vor B0/A2 festgelegt werden.

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

CLI-Optionen überschreiben das Preset. In der UI können Parameter ebenfalls
angepasst werden; solche Anpassungen müssen für einen Vergleich aufgezeichnet
werden. Eigene ReID-Presets liegen in `data/modes/reid_presets.json`.
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
ausgewählten Matches und Updates; sie sind kein vollständiger Frame-Export.

Standardpfade:

- `data/input/`: Eingabevideos
- `data/db/reid.sqlite3`: Profile, Ereignisse und Läufe
- `data/snapshots/`: Personenausschnitte
- `data/output/`: annotierte Videos

Pfade lassen sich über die Variablen aus `.env.example` einstellen. Synthetische
IDs machen erkennbare Aufnahmen nicht anonym. Eingaben, Datenbanken und generierte
Ausgaben gehören nicht in Git.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Die Regressionstests laufen ohne Kamera und ohne Modell-Downloads. Sie prüfen
Ressourcenfreigabe, Uploads, Pfadschutz, Presets, Datenhaltung und die Trennung von
Analysebildern und Annotation. Ein Modell-Smoke-Test und die quantitative
Evaluation sind davon getrennte Prüfungen.

## Reset

```powershell
python scripts/reset_db.py
```

Das löscht die konfigurierte Datenbank einschließlich WAL/SHM, Snapshots und
Ausgabevideos. Nur Ziele unterhalb von `data/` werden akzeptiert; `data/` selbst
und Ziele außerhalb werden abgewiesen. Vor einem Reset benötigte Versuchsdaten
sichern. Für Experimentisolation eigene Pfade verwenden.
