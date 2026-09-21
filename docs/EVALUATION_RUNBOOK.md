# Reduzierte Evaluation: vom Pilot zum Ereignisvergleich

Stand: 20. September 2026. Dieses Runbook dokumentiert den ursprünglich geplanten
größeren Versuch und bleibt als Anleitung für Erweiterungen erhalten. Tatsächlich
wurden je ein Video aus G1–G4 mit B0/A2/A3 verarbeitet, also zwölf Läufe. Die
Ergebnisse stehen in [four_group_results.md](evaluation/four_group_results.md).
Die Auswertung ist eine Fallstudie, kein umfassender Detektor-/Trackingbenchmark.

## Verbindlicher Umfang

| Gruppe | Aufnahmeinhalt | Getrennte Testaufnahmen |
|---|---|---|
| G1 | Freie Sicht, mehrere Personen, Bewegung, unbekannter Eintritt | 3 |
| G2 | Registrierung, Verlassen und Rückkehr mit neuem Track | 3 |
| G3 | Ähnliche Kleidung mit mehreren Personen und Rückkehr | 3 |
| G4 | Kreuzung/Verdeckung mit einem kritischen Übergang je Aufnahme | 3 |

Der ursprüngliche Zielumfang waren zwölf Testsequenzen à etwa 30–60 Sekunden,
jeweils B0/A2/A3: **36 Läufe**. Dieser Umfang wurde nicht erhoben.
Eine Registrierung-/Rückkehr-Videopaarung zählt als eine Sequenz. Pilotaufnahmen
und zusätzliche technische Laufzeitwiederholungen zählen nicht zu den zwölf.
Ein externer Clip und ein Leerraum-Negativtest sind optional, getrennt zu berichten.
Weitere Tracker, YOLO-/OSNet-Architekturen, Beleuchtungs- oder Kleidungswechsel
gehören nicht zum Kernvergleich. Zusätzliche Aufnahmen derselben Personen prüfen
Robustheit auf diesen Aufnahmen, nicht Generalisierung auf unbekannte Personen.

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

## 2. Im Pilot eine Konfiguration wählen, nicht alles kombinieren

Separate Pilotaufnahmen enthalten Rückkehr, ähnliche Kleidung und Verdeckung.
Die Testaufnahmen bleiben unberührt; benachbarte Frames desselben Videos nicht
auf beide Mengen verteilen. Auch Pilotversuche mit geänderten Parametern bekommen
jeweils eine frische Versuchsdatenbank, keine bereits optimierten Profile.

Konstant lassen: YOLOv8n, ByteTrack samt YAML, Detektionskonfidenz 0,35,
Eingangsgröße 640, fünf akzeptierte Initialbeobachtungen,
Updateintervall fünf Frames,
Mindestcropgröße 30 × 80 Pixel, Padding 0,05, Qualitätsformel und Gewichtung.
Das sind Rahmenbedingungen, keine als optimal nachgewiesenen Werte.

Begrenztes Ausgangsraster für OSNet:

1. Matching: 0,75 / 0,80 / 0,82 / 0,85 / 0,90; andere Werte zunächst wie B0.
2. Mit dem ausgewählten Matchingwert Update-Ähnlichkeit: 0,80 / 0,85 / 0,90.
3. Mit diesen Werten Qualitätspaare (Initialisierung/Update):
   0,45/0,55; 0,55/0,65; 0,65/0,75.
4. Die gewählte Kombination gemeinsam nochmals prüfen.

Das ist eine schrittweise Suche, nicht das Produkt aller Parameterkombinationen.

Auswahl vorher festlegen: zunächst die wenigsten falschen Personen-ID-Zuordnungen,
bei Gleichstand die meisten korrekten Rückkehr-/unbekannten Eintrittsentscheidungen,
danach die kürzere mediane Zeit bis zur korrekten ID. Konfigurationen ohne eine
korrekte Rückkehr sind keine brauchbare Wahl; wenn alle scheitern, erst die Ursache
prüfen statt eine finale Konfiguration zu behaupten. Immer fehlende Entscheidungen,
Fehlregistrierungen und falsche neue IDs zusätzlich berichten. Den Update-Schutz
an falschen akzeptierten und korrekten abgelehnten Updates in den annotierten
Übergangsfenstern prüfen. Kein falscher Updatekandidat bedeutet keinen Nachweis
eines Schutzvorteils.

Je Pilotkonfiguration eine Ergebniszeile festhalten: Preset-ID, sämtliche Parameter,
Clip-/Ereignisbestand, richtige/falsche/neue/fehlende IDs, Registrierungsmisserfolge,
Updatebefunde mit Nennern und Zuweisungszeit. Für die ausgeführte Evaluation
wurden Matching und Update-Ähnlichkeit auf 0,75 sowie die Qualität auf 0,55/0,65
festgelegt.

## 3. Drei eigene finale Presets einfrieren

| Variante | Qualität | Matching-/Update-Ähnlichkeit |
|---|---|---|
| B0 | Gewählte Pilotgrenzen | Gewählte OSNet-Werte |
| A2 | Alle Qualitätsgrenzen 0 | Wie B0 |
| A3 | Wie B0 | Matching wie B0; Update-Ähnlichkeit -1 |

In der UI die ausgewählte OSNet-Konfiguration als `eval_b0` speichern. Jeweils
**dieses gespeicherte B0 laden**, ändern und als neues Preset speichern:

- `eval_a2`: `min_embedding_quality`, `min_initial_blur_score`,
  `min_border_blur_score`, `min_initial_aspect_ratio_score` und
  `min_update_quality` auf 0 setzen.
- `eval_a3`: nur `min_update_similarity` auf -1 setzen.

A2 behält Mindestgrößen, Qualitätsgewichtung, Initialpuffer und
Snapshot-Auswahl. A3 lässt Qualitätsgrenzen und Profilupdates aktiv; es deaktiviert
nur die Ähnlichkeitsprüfung. Nicht irrtümlich Updates vollständig deaktivieren.

Die eingebauten Presets `default`, `no_quality_thresholds` und
`no_update_similarity` sind Ausgangskonfigurationen. Sie übernehmen **nicht**
automatisch die im Pilot bearbeiteten Werte. Deshalb explizit die drei gespeicherten
finalen Presets verwenden und ihre Parameter vor dem Test vergleichen.
Code, Gewichte, YAML, Hardware und Presetdatei festhalten; keine Nachkalibrierung
auf Testvideos. Die konkrete Quelle ist ein Laufparameter, kein Presetparameter.

## 4. Ereignisse annotieren und Entscheidungen prüfen

Für echte Personen `GT_01` bis `GT_04` verwenden. Vor Sichtung der Vorhersagen
Registrierungsphase, unbekannte Eintritte, Austritt und ersten wieder sichtbaren
Frame markieren. Je G4-Aufnahme genau ein dreisekündiges Übergangsfenster wählen:
0,5 Sekunden vor bis 2,5 Sekunden nach dem markierten kritischen Übergang.
Ein zweites Projektmitglied prüft schwierige Zuordnungen. Nicht alle Frames mit
Bounding Boxes annotieren.

Für jede Variante die richtige, eindeutige Zuordnung zwischen GT-Person und
ursprünglicher System-ID in der Registrierung prüfen. Numerische System-IDs nicht
zwischen Varianten vergleichen. Je Rückkehr festhalten:

- GT-/Ereignis-ID, Clip, Austritt und Rückkehrframe/-zeit.
- Ursprüngliche System-ID oder fehlgeschlagene Erstregistrierung.
- Erste ausgegebene Personen-ID innerhalb von zwei Sekunden und Entscheidungszeit.
- Kategorie: richtige alte ID, falsche bestehende ID, neue ID oder keine Entscheidung;
  fehlgeschlagene Erstregistrierung als eigener Misserfolg.
- Personen-ID am Ende des zweisekündigen Fensters als Stabilitätskontrolle.
- Alter/neuer/fehlender Track und gegebenenfalls Fehlerursache.

Die Haupt-Erfolgsrate ist `erste rechtzeitige alte ID korrekt / alle auswertbaren
Rückkehrereignisse`. Zusätzlich die Rate für zuvor richtig registrierte Personen,
für Rückkehr mit Track-Neustart oder ohne erhaltenen Track und die
Stabilitätskontrolle mit jeweiligen Nennern
berichten. Keine anfängliche falsche Entscheidung rückwirkend durch spätere
Korrekturen ersetzen. Die Zeit bis zur korrekten ID nur zusammen mit der Erfolgsrate
und den Misserfolgen angeben.

Sichtbarkeitsausschlüsse allein aus GT vorab bestimmen und zählen. Das gesamte
Zweisekundenfenster muss anhand des Videos auswertbar sein. Fehlende Detektion,
fehlender Track, abgelehnte Crops oder fehlgeschlagene Registrierung sind dagegen
Systemfehler, kein Grund zum Entfernen aus dem Nenner. Unbekannte Eintritte sind
korrekt, wenn rechtzeitig eine neue, keiner registrierten Person gehörende ID entsteht.

In den G4-Fenstern alle protokollierten Profilupdateversuche anhand von Crop,
zeitlichem Kontext im Referenzvideo und registriertem Profileigentümer prüfen.
Für strittige Zuordnungen ist die Prüfung durch das zweite Projektmitglied
vorgesehen. Bleibt etwa bei Verdeckung oder mehreren plausiblen Personen die
Zuordnung ungeklärt, den Crop als uneindeutig separat zählen und nur aus den
Anteilsnennern dieser Profilupdate-Teilanalyse ausnehmen. Zählen: falsche akzeptierte
Updates / akzeptierte auswertbare Versuche und korrekte abgelehnte Updates /
abgelehnte auswertbare Versuche. Fenster ohne Updateversuche separat ausweisen;
Nenner 0 heißt nicht definiert (`n. d.`), nicht 0 % Fehler. Zusätzlich die falschen
Personenzuordnungen an diesen Übergängen beschreiben. Ein geschütztes Profil
repariert nicht automatisch eine falsche Tracker-/Personenzuordnung.

Die Aussagen gelten für diese Ereignisse/Fenster. Keine vollständigen Precision-/
Recall-, IDF1- oder ID-Switch-Zahlen daraus ableiten. TrackEval und dichte
Referenztrajektorien sind keine Pflicht dieser Kernevaluation. Pro Clip und Gruppe
Rohzahlen sowie paarweise B0/A2/A3-Ergebnisse berichten; bei wenigen Personen keine
breite Generalisierung oder unabhängige Stichproben aus einzelnen Frames behaupten.

## 5. Unabhängige Versuchseinheit starten

```powershell
python -m app.evaluation --sources data/input/g2_take1.mp4 --modes eval_b0 eval_a2 eval_a3 --device cpu
```

Die drei `eval_*`-Presets müssen zuvor in der UI gespeichert sein. Ein Aufruf
verarbeitet eine Sequenz mit den drei Varianten einmal, mit vollständigem Clip
(`max_frames=0`) und neuer Datenbank pro Variante. Für alle zwölf Sequenzen separat
aufrufen. Ohne `--modes` laufen die drei eingebauten, noch nicht kalibrierten Presets.
`--repetitions 3` macht aus einem Video keine drei getrennten Aufnahmen.
Mehrere **unabhängige Szenarien separat aufrufen**, nicht als gemeinsame Quellenliste.

Für eine zusammengehörige Registrierung-/Rückkehrpaarung:

```powershell
python -m app.evaluation --sources data/input/registrierung.mp4 data/input/rueckkehr.mp4 --modes eval_b0 eval_a2 eval_a3 --device cpu
```

Die Quellen werden in dieser Reihenfolge verarbeitet und teilen die Datenbank
nur innerhalb ihrer Versuchseinheit. Jeder Clip bekommt eine neue Pipeline samt
Tracker. Keine Pipelineinstanz zwischen Clips wiederverwenden.

In der UI ist die neue Datenbank standardmäßig aktiv. Das Abschalten teilt bewusst
den interaktiven Profilbestand des gewählten Encoders/Checkpoints. Die normale
`python -m app.main`-CLI verwendet ebenfalls dessen gemeinsamen Bestand unter
`data/db/encoders/<encoder_key>/reid.sqlite3`; sie ist kein isolierter Vergleichslauf.
Andere Encoder erhalten andere Datenbanken. Zurückwechseln verwendet den
ursprünglichen Bestand. Altbestände werden nicht automatisch übernommen.

## 6. Artefakte kontrollieren

Unter `data/experiments/<unit_id>/` liegen:

- `experiment.json`: Reihenfolge der Quellen, Konfiguration, Profilpolitik und Gesamtstatus.
- `db/encoders/<encoder_key>/reid.sqlite3`: Profile/Ereignisse, nur für diese Einheit und Encoder-Konfiguration.
- `snapshots/<run_id>/<person_id>/frame_*.jpg`: kollisionsfreie Diagnosebilder.
- `output/*.mp4`: annotierte Ausgabevideos.
- `output/runs/<run_id>/manifest.json`, `frames.jsonl`, `tracking_mot.txt`.

`frames.jsonl` enthält genau einen Datensatz pro erfolgreich verarbeitetem Frame:

```json
{"frame_index": 4, "timestamp_seconds": 0.12, "detections": [{"bbox_xyxy": [2, 5, 25, 44], "confidence": 0.9, "track_id": 1, "person_id": "person_000001", "state": "created_identity", "decision_frame_index": 4, "snapshot_frame_index": 2}]}
```

Weitere Felder: `run_id`, Klasse, Qualität, Match-Score und gegebenenfalls
`profile_update_accepted`, `update_similarity`, `profile_update_reason`. Unbekannte Kennungen
sind JSON `null`. Ein leerer Frame enthält `"detections": []`. Anfangs ausbleibende
Personen-IDs bleiben bestehen; keine rückwirkende Korrektur im Export. Der
Ereignisframe in SQLite ist jetzt der Entscheidungsframe; `snapshot_frame_index`
hält den möglicherweise früheren Diagnoseframe fest.

Die MOT-Datei enthält `frame,track_id,x,y,w,h,confidence,-1,-1,-1` und nur echte
Trackerkennungen. Sie bleibt für spätere dichte Trackingauswertungen verfügbar,
wird für die aktuelle Ereignisevaluation aber nicht benötigt. Die GT-Auswertung
mit ursprünglichen Videos und den Exporten erfolgt manuell und ist nicht Teil des
Versuchstarters. Für den Vier-Video-Lauf wurde sie mit
`scripts/evaluate_four_group_results.py` aus der geprüften Ereignisreferenz erzeugt.
Ungetrackte Boxen und fehlende Personen-IDs bleiben sichtbar.

Frameindices beginnen bei 1. Zeitstempel sind nominal `(frame_index-1)/fps`.
OpenCV liefert hier keine ursprünglichen PTS variabler Frameraten; für zeitliche
Evaluation Clips mit konstanter/überprüfter FPS verwenden. FPS-Fallbacks stehen
im Manifest. Kameraquellen können nicht als Datei gehasht werden.

## 7. Status und Timing prüfen

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

Nur eine vorab festgelegte repräsentative Sequenz je finalem Preset dreimal
technisch verarbeiten, um Laufzeitschwankungen abzuschätzen. Nicht alle zwölf
Sequenzen dreimal ausführen. Diese Wiederholungen separat von den 36 Kernläufen
ausweisen und nicht als unabhängige Erkennungsversuche zählen.

## Erweiterungen nach der Fallstudie

Für belastbarere Aussagen sind zusätzliche, zurückgehaltene Testsequenzen mit mehr
Personen und gezielt auftretenden Fremdkandidaten in den Updatefenstern erforderlich.
Ein optionales öffentliches Video braucht dokumentierte Lizenz/Berechtigung. Eine
vollständige Trackingbewertung würde außerdem dichte Referenztrajektorien benötigen.
Profilverunreinigung nach Tracker-ID-Switches bleibt trotz Ähnlichkeitsschutz eine
mögliche Methodengrenze.

Die Profile akkumulieren jetzt rohe qualitätsgewichtete Summen inklusive aller
Initial-Crops. Updates benötigen zusätzlich `min_update_similarity` (in der
Auswertung 0,75). Ablehnungen werden exportiert und verändern
das Profil nicht. Sie reparieren die Tracker-/Personenzuordnung nicht automatisch.
Formeln und Qualitätsberechnung: [PROFILE_UPDATES.md](PROFILE_UPDATES.md).
