# Evaluationsumfang auf main

Stand: 12. September 2026. Dieser Branch enthält den abgegrenzten ReID-Prototyp.
Die quantitative Evaluation wurde noch nicht durchgeführt.

## Enthalten und dokumentiert

- Videos und lokale Webcam als Quelle.
- YOLO-Personendetektion mit ByteTrack; BoT-SORT bleibt eine auswählbare Alternative.
- Person-Crops, geometrische Mindestgrößen und heuristische Qualitätsbewertung.
- Drei gute Initialbeobachtungen, qualitätsgewichtete Embeddings und Profilupdates.
- OSNet-Adapter und Farbhistogramm als Vergleichsencoder.
- Cosine Matching, synthetische Personen-IDs und SQLite.
- Annotierte Videos, Vorschau, CLI und ReID-Presets in Streamlit.

## Presets und Methodenabgrenzung

| Preset | Konfiguration | Unterschied zu B0 |
|---|---|---|
| `default` | B0 | Referenz: OSNet, Schwellen 0,55 / 0,65 |
| `colorhist` | A1 | Nur der Encoder wird ersetzt |
| `no_quality_thresholds` | A2 | Nur die beiden Annahmeschwellen werden null |

A2 behält Qualitätsgewichtung, Mindestgrößen, Initialpuffer und Snapshot-Auswahl.
Alle drei Presets enthalten zunächst dieselben sonstigen Parameter. CLI/UI-Overrides
sind möglich und müssen als Konfigurationsänderung aufgezeichnet werden.

Die UI lädt Presets in einen gemeinsamen, vollständigen Pipeline-Editor. Starten
und "Aktuelle Einstellungen speichern" verwenden dieselben Parameter. Geänderte
Läufe werden im Namen markiert; die effektive Pipeline-Konfiguration ist einsehbar
und wird vollständig in `analysis_runs.metadata_json` festgehalten. Das Speichern
erstellt ein neues Preset und lädt es; vorhandene Presets werden nicht überschrieben.
Alle vier konfigurierbaren Matching-/Konfidenz-/Qualitätsschwellen, Mindestgrößen
und zeitlichen Parameter sind editierbar. Tracker-interne Schwellen bleiben in der
Tracker-YAML, Konstanten und Gewichte der Qualitätsheuristik bleiben unverändert.
Diese Einstellbarkeit dient Pilotversuchen, nicht einer Nachkalibrierung auf Testclips.

Die bisherige zusätzliche Bewegungsdiagnostik ist vollständig aus dem aktiven
Hauptpfad entfernt. Das betrifft Richtungsvektoren, Pixelgeschwindigkeit,
Sprung-Warnungen, Payload-Felder und eigene UI-Steuerung. Die interne Bewegungsschätzung
des verwendeten Trackers bleibt Teil des Trackings. UC-06 (Gehen und Richtungswechsel)
bleibt deshalb als Versuchssituation sinnvoll; eine eigene Motion-Metrik wird nicht behauptet.

Fußballmodule, Team-/Ball-/Spielfeldmodelle, ihre Tabellen-Erzeugung und die ungenutzte
Qdrant-Servicekonfiguration liegen nur noch im gesicherten Entwicklungsstand.

## Wiederherstellbarer Entwicklungsstand

`codex/research-extensions` am Commit `e8a0ba8` bewahrt den bisherigen Code,
die ursprüngliche LaTeX-Dokumentation und deren Archivfassungen. Auch
`Frames_Optimization` sowie die fünf bisherigen Fix-Branches bleiben erhalten.
Der Branch kann für spätere Forschung ausgecheckt werden, ohne den ReID-Kern zu erweitern.

Historische Datenbanken, Bilder und Videos wurden nicht gelöscht. Beim Öffnen alter
Datenbanken bleiben zusätzliche Tabellen erhalten. Auf diesem Branch neu angelegte
Datenbanken enthalten nur `persons`, `events` und `analysis_runs`.
Neue benutzerdefinierte Presets verwenden `data/modes/reid_presets.json`;
die bisherige `custom_modes.json` wird weder geladen noch überschrieben.

## Übernommene Korrekturen

- Ressourcen werden auch im Fehlerpfad freigegeben; Writer und Framegrenze werden geprüft.
- SQLite-Verbindungen werden nach Commit beziehungsweise Rollback geschlossen.
- Reset-Ziele werden vor dem Löschen validiert.
- Wiederholte Uploads verwenden dieselbe Datei; Dateinamen werden bereinigt.
- Analysebilder und Annotation sind getrennt, damit Markierungen keine Crops verändern.
- Sichtbare bekannte Personen-IDs werden vor der Bearbeitung neuer Tracks reserviert.
- Preset-Namen werden korrekt in die Laufkonfiguration übernommen.

Der bisherige Identitäts-Fix wurde nicht vollständig übernommen: Die darin enthaltene
Motion-basierte Neuzuordnung würde eine andere Methode einführen. Bekannte Tracks
behalten weiterhin ihre Personenkennung; qualitätsgefilterte Updates prüfen ihre
Identität noch nicht erneut. Dies ist im Paper als Einschränkung dokumentiert.

## Die sechs technischen Vorbereitungen sind implementiert

1. OSNet-x1.0 nutzt veröffentlichte MSMT17-ReID-Gewichte, ohne eigenes Training.
   Download/Herkunft/Prüfsumme sind in [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) festgehalten.
   Echter Encoder-Smoke-Test erfolgreich; keine stillen ImageNet-/Teilinitialisierungs-Fallbacks.
2. Fehlende Trackerkennungen bleiben null. Keine erfundenen Ersatz-IDs, keine
   Profilbildung mit instabilen IDs; ungetrackte Boxen bleiben im Frame-Export.
3. `frames.jsonl` exportiert alle ausgegebenen Boxen und leere Frames. Tracking-
   und Personenkennung bleiben getrennt, Entscheidungs-/Snapshotframe ebenfalls.
   Keine rückwirkende Personen-ID-Zuweisung. `tracking_mot.txt` enthält echte Track-IDs.
4. `python -m app.evaluation` erzeugt getrennte Versuchseinheiten pro Variante/
   Wiederholung und frische Tracker pro Quelle. Nur zusammengehörige Quellen teilen
   Profile. UI-Läufe verwenden standardmäßig eine neue Datenbank.
5. Snapshots liegen unter Laufkennung/Personenkennung; Schreibfehler werden erkannt.
6. Laufmanifest mit vollständiger Konfiguration, tatsächlichen Datei-/Gewichte-/
   Tracker-Hashes, Codecommit/Dirty-Status/Quellcode-Hash, Paketversionen, Hardware,
   Zeitmessungen, verarbeiteten Frames und abgeschlossenem/fehlgeschlagenem Status.
   Die lokale technische Umgebung ist als Versionssnapshot festgehalten.

## Vor der quantitativen Evaluation noch erforderlich

Annotierte Pilot-/Testclips samt Berechtigung/Lizenz auswählen, Metriken gegen
Ground Truth berechnen, Rückkehrentscheidungen prüfen, danach Code und Parameter
einfrieren. Bedienung/Format: [EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md).
Der Versuchsstarter führt noch keine Ground-Truth-/TrackEval-Auswertung durch.

Weiterhin bekannte Grenzen: kein expliziter Ablauf alter Track-Zustände,
mögliche Profilverunreinigung nach ID-Switches, keine Encoder-Versionsprüfung
bei gleicher Dimension im bewusst geteilten interaktiven Bestand, nominale
FPS-Zeitstempel statt ursprünglicher VFR-PTS. Modellwechsel mit neuer Datenbank
auswerten. Profilupdate-Änderungen konsistent vor dem Einfrieren der Methode entscheiden.

## Prüfungen für diesen Branch

`python -m unittest discover -s tests -v` prüft den ReID-Umfang und die konkreten
Korrekturen mit temporären Daten und Fake-Komponenten. Zusätzlich laufen opt-in
echte Modell-/Video-Tests für B0/A1/A2 mit künstlichen Clips ohne Personen.
Beide echten Tests sind lokal erfolgreich. Damit wird weder eine
ReID-Genauigkeit noch eine Echtzeitfähigkeit nachgewiesen. Das Paper enthält dafür
weiterhin den Versuchsplan und noch nicht erhobene Ergebnisse.
