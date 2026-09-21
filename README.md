# Local Person Re-Identification - Masterprojekt

`main` enthält den ReID-Kern für die dokumentierten Projektversuche:
Video oder Webcam, YOLO-Personendetektion, ByteTrack/BoT-SORT, Person-Crops,
Qualitätsprüfung, OSNet, synthetische Personen-IDs und SQLite.

**Start für Mitwirkende:** [Codebase Guide](docs/CODEBASE_GUIDE.md).
**Eigene Tracker und Verfahren:** [Backend-Schnittstellen](docs/EXTENDING_BACKENDS.md).
**Profilrechnung und Profilschutz:** [Personenprofile](docs/PROFILE_UPDATES.md).
**Umfang und offene Voraussetzungen:** [Evaluationsstand](docs/EVALUATION_SCOPE.md).
**Versuchsplan und Paper:** [LaTeX-Quelle](docs/technische_systemdokumentation.tex).
**Vier vorhandene Gruppenclips auswerten:** [Vier-Video-Evaluation](docs/FOUR_VIDEO_EVALUATION.md).

Die explorative Vier-Video-Evaluation ist abgeschlossen. Vier Gruppenclips wurden
mit B0, A2 und A3 in zwölf isolierten Läufen verarbeitet und ereignisbasiert
ausgewertet. B0 und A3 erkannten jeweils 5/6 Rückkehrereignissen korrekt, A2 3/6.
Versuchsaufbau, Resultate und Grenzen stehen im Paper; die reproduzierbare
Kurzfassung liegt unter [Evaluationsresultate](docs/evaluation/four_group_results.md).

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
| `default` | B0 | OSNet | 0,55 / 0,65; Schärfe 0,40 / 0,45; Personenformat 0,50 |
| `no_quality_thresholds` | A2 | OSNet | alle Qualitätsgrenzen 0 |
| `no_update_similarity` | A3 | OSNet | 0,55 / 0,65; Schärfe 0,40 / 0,45; Personenformat 0,50 |

Die eingebauten Ausgangspresets verwenden zunächst YOLOv8n, ByteTrack,
Cosine-Schwellwert 0,82, Detektionskonfidenz 0,35, Eingangsgröße 640, fünf
Initialbeobachtungen im Abstand von drei Frames und Updates alle zehn Frames.
Die Mindest-Crop-Größe beträgt 30 x 80 Pixel. Für die berichtete Evaluation wurde
das gespeicherte Preset `best-calibrated` mit Matching- und Updateschwelle 0,75
sowie einem Updateintervall von fünf Frames verwendet.
A2 behält Mindestgrößen, Qualitätsgewichtung und Snapshot-Auswahl bei.
A3 verändert nur `min_update_similarity` auf -1: Updates und Qualitätsgrenzen
bleiben aktiv, die zusätzliche Ähnlichkeitsprüfung ist aus. Die anderen Presets
verwenden zunächst 0,82 für diese Prüfung. Das sind Ausgangs-, keine finalen Pilotwerte.

Für einen vollständigen Clip ausdrücklich `--max-frames 0` verwenden; der
interaktive Standard begrenzt den Lauf auf 500 Frames.

```powershell
python -m app.main --source data/input/pilot.mp4 --mode default --max-frames 0
python -m app.main --source data/input/pilot.mp4 --mode no_quality_thresholds --max-frames 0
```

Diese Befehle verwenden einen geteilten Bestand **pro OSNet-Konfiguration**.
Andere Architekturen oder Checkpoint-Inhalte erhalten andere Datenbanken. Unabhängige
Versuchseinheiten brauchen getrennte Ausgangszustände und Pfade; die Befehle
allein stellen noch keinen isolierten Vergleichslauf her.

Für weitere isolierte Versuche den Versuchsstarter nutzen:

```powershell
python -m app.evaluation --sources data/input/pilot.mp4
```

Das verarbeitet das vollständige Video mit den drei eingebauten Ausgangspresets,
je Variante mit neuer Datenbank. Die tatsächlich berichtete Fallstudie verwendete
je einen Clip aus vier Versuchsgruppen und damit zwölf Läufe. Der ursprünglich
größer geplante Bestand mit zwölf Sequenzen und 36 Läufen wurde nicht erhoben.
Eine weitere Sequenz lässt sich beispielsweise so starten:

```powershell
python -m app.evaluation --sources data/input/g2_take1.mp4 --modes eval_b0 eval_a2 eval_a3 --device cpu
```

Die `eval_*`-Presets sind selbst zu speichern; eingebaute Varianten übernehmen
nicht automatisch bearbeitete B0-Werte. Bei Bedarf können zusammengehörige
Registrierungs-/Rückkehrclips gemeinsam unter `--sources` angegeben werden.
Alle übergebenen Quellen gehören **einer** Versuchseinheit an und teilen deren
Profile; pro Quelle wird trotzdem ein frischer Tracker verwendet. Unabhängige
Szenarien separat starten. Anleitung und Exportformat: [EVALUATION_RUNBOOK.md](docs/EVALUATION_RUNBOOK.md).
In der UI ist "Isolierter Lauf (neue Datenbank)" standardmäßig aktiv.

CLI-Optionen überschreiben das Preset. In der UI können Parameter ebenfalls
angepasst werden. Es gibt einen gemeinsamen Editor: Preset laden, Parameter
bearbeiten und entweder starten, die aktuellen Werte als neues Preset speichern
oder ein ausgewähltes eigenes Preset aktualisieren.
Alle Laufzeitparameter aus `PipelineConfig` haben genau ein Eingabefeld. Dazu
gehören alle konfigurierbaren Konfidenz-, Matching-, Qualitäts- und Überlappungsschwellen,
Crop-Mindestgrößen, Padding und zeitliche Parameter. Schwellen lassen sich auch
als genaue Dezimalwerte eingeben. Interne Tracker-Schwellen gehören zur gewählten
Tracker-YAML; die festen Konstanten der Qualitätsheuristik werden nicht verändert.
Die UI erlaubt Detektionskonfidenz ab 0,0001, nicht exakt null: Ultralytics würde
den Nullwert beim Tracking durch seinen Standardwert 0,1 ersetzen.

Änderungen überleben normale UI-Neuausführungen. Ein Presetwechsel lädt dessen
gespeicherte Werte; "Änderungen verwerfen / Preset neu laden" setzt den Editor zurück.
Ein veränderter Lauf wird als "(geändert)" markiert. Die effektive Konfiguration
ist vor dem Start einsehbar und wird vollständig in den Laufmetadaten gespeichert.
Nach dem Speichern wird das Preset automatisch geladen. Referenzpresets können
nicht überschrieben werden; für eigene Presets gibt es eine ausdrücklich
beschriftete Aktualisierungsaktion bei unveränderter Preset-ID.

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
- `data/db/encoders/<encoder_key>/reid.sqlite3`: Profile, Ereignisse und Läufe pro Encoder-Konfiguration
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
Analysebildern und Annotation. Ein Modell-Smoke-Test und die ausgeführte
Vier-Video-Evaluation sind davon getrennte Prüfungen.

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
