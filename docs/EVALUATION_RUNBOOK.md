# Vom Pilotclip zum reproduzierbaren Versuch

## 1. Umgebung prüfen

Die lokal technisch geprüften Versionen stehen in `requirements-evaluation-lock.txt`.
Das ist ein Versionssnapshot für Windows/Python 3.13, keine pauschale CUDA-Garantie.
Bei anderer Hardware/Python-Version den Smoke-Test wiederholen. Gewichte gemäß
[MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) einrichten.

```powershell
python -m pip check
python -m app.evaluation.smoke --device cpu
python -m unittest discover -s tests -v
$env:REID_REAL_SMOKE = '1'
python -m unittest tests.test_real_pipeline_smoke -v
Remove-Item Env:REID_REAL_SMOKE
```

## 2. Pilot und Test trennen

Pilotclips zum Einstellen der Schwellen verwenden und als eigenes UI-Preset
speichern. Danach Codeversion, Gewichte und Parameter einfrieren. Testclips mit
Referenzboxen und echten Personenkennungen annotieren; diese Kennungen sind nicht
die vom System erzeugten Personen-IDs. Ein-/Austritts- und Rückkehrframes markieren.
Keine Genauigkeitszahl allein aus der Zahl erzeugter Profile ableiten.

Die eigene Qualitätsheuristik und Mindestgrößen beeinflussen die Stichprobe.
A2 entfernt nur die beiden Qualitätsschwellen, nicht alle Qualitätsmechanismen.
Keine Tracker-ID ist ein anderer Zustand als keine Personen-ID; beides auswerten.

## 3. Unabhängige Versuchseinheit starten

```powershell
python -m app.evaluation --sources data/input/test.mp4 --repetitions 3 --device cpu
```

Standardmäßig B0/A1/A2, vollständiger Clip (`max_frames=0`), neue Datenbank pro
Variante und Wiederholung. Für ein gespeichertes Pilotpreset `--modes mein_pilot`.
Mehrere **unabhängige Szenarien separat aufrufen**.

Für Registrierung und Rückkehr über zwei Videos (UC-12):

```powershell
python -m app.evaluation --sources data/input/registrierung.mp4 data/input/rueckkehr.mp4 --modes default --repetitions 3
```

Die Quellen werden in dieser Reihenfolge verarbeitet und teilen die Datenbank
nur innerhalb ihrer Versuchseinheit. Jeder Clip bekommt eine neue Pipeline samt
Tracker. Keine Pipelineinstanz zwischen Clips wiederverwenden.

In der UI ist die neue Datenbank standardmäßig aktiv. Das Abschalten teilt bewusst
den interaktiven Profilbestand. Die normale `python -m app.main`-CLI behält ebenfalls
den bisherigen gemeinsamen Datenbankpfad; sie ist kein isolierter Vergleichslauf.

## 4. Artefakte kontrollieren

Unter `data/experiments/<unit_id>/` liegen:

- `experiment.json`: Reihenfolge der Quellen, Konfiguration, Profilpolitik und Gesamtstatus.
- `db/reid.sqlite3`: Profile/Ereignisse, nur für diese Einheit.
- `snapshots/<run_id>/<person_id>/frame_*.jpg`: kollisionsfreie Diagnosebilder.
- `output/*.mp4`: annotierte Ausgabevideos.
- `output/runs/<run_id>/manifest.json`, `frames.jsonl`, `tracking_mot.txt`.

`frames.jsonl` enthält genau einen Datensatz pro erfolgreich verarbeitetem Frame:

```json
{"frame_index": 4, "timestamp_seconds": 0.12, "detections": [{"bbox_xyxy": [2, 5, 25, 44], "confidence": 0.9, "track_id": 1, "person_id": "person_000001", "state": "created_identity", "decision_frame_index": 4, "snapshot_frame_index": 2}]}
```

Weitere Felder: `run_id`, Klasse, Qualität und Match-Score. Unbekannte Kennungen
sind JSON `null`. Ein leerer Frame enthält `"detections": []`. Anfangs ausbleibende
Personen-IDs bleiben bestehen; keine rückwirkende Korrektur im Export. Der
Ereignisframe in SQLite ist jetzt der Entscheidungsframe; `snapshot_frame_index`
hält den möglicherweise früheren Diagnoseframe fest.

Die MOT-Datei enthält `frame,track_id,x,y,w,h,confidence,-1,-1,-1` und nur echte
Trackerkennungen. Für IDF1/ID-Switches der Tracker diese Ansicht verwenden. Für
Precision/Recall der kombinierten Erfassung **alle Boxen aus JSONL** einbeziehen,
auch ungetrackte. Wiedererkennung anhand `person_id` getrennt auswerten. Die
Metrikberechnung gegen Ground Truth ist noch nicht Teil des Versuchstarters.

Frameindices beginnen bei 1. Zeitstempel sind nominal `(frame_index-1)/fps`.
OpenCV liefert hier keine ursprünglichen PTS variabler Frameraten; für zeitliche
Evaluation Clips mit konstanter/überprüfter FPS verwenden. FPS-Fallbacks stehen
im Manifest. Kameraquellen können nicht als Datei gehasht werden.

## 5. Status und Timing prüfen

Ein fertiger Versuch braucht `status=completed` sowohl im Einheiten- als auch im
Laufmanifest. Fehler bleiben als `failed` mit Fehlertyp und Meldung sichtbar;
partielle Frame-Exporte nicht als vollständige Messungen verwenden. Keine dekodierbaren
Frames oder ein vor der deklarierten Framezahl endendes Datei-Video gelten als Fehler.
Bei unzuverlässiger Container-Framezahl die Datei vor dem Versuch korrigieren.

Das Manifest hält vollständige Parameter, Eingabe-/Detektor-/Checkpoint-/Tracker-
Hashes, Codecommit, Dirty-Status und tatsächlichen Quellcode-Hash, Paketversionen,
Python/OS/CPU/GPU sowie verarbeitete Frames fest. Modellladezeit wird von der
Verarbeitung getrennt. Verarbeitung umfasst I/O, Modelle, Datenbank, Exporte,
Videoausgabe, Callbacks und Ressourcenfreigabe; initiale Hash-/Manifestvorbereitung
ist nicht darin enthalten. FPS und Real-Time-Faktor beruhen auf dieser Laufzeit.
UI-Vorschau für Laufzeitvergleiche identisch einstellen oder CLI ohne UI nutzen.

## Was noch zu tun ist

Annotierte Pilot-/Testclips auswählen, Lizenz/Berechtigung des öffentlichen Videos
festhalten, Ground-Truth-Auswertung und TrackEval-Einbindung ergänzen, Fehlerfälle
visuell kontrollieren und erst dann quantitative Ergebnisse in das Paper eintragen.
Profilverunreinigung nach Tracker-ID-Switches bleibt eine Methodengrenze.
