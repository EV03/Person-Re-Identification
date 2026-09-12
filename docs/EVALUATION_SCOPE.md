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

## Vor der abschließenden Evaluation noch erforderlich

1. Einen expliziten, überprüften OSNet-Checkpoint sowie Paket-/Tracker-Versionen festlegen.
   Der aktuelle Adapter verwendet `model_path=""`; die Presets garantieren keine
   ReID-trainierten Gewichte. Die Torchreid-Paketstruktur muss im Installations-Smoke-Test passen.
2. Alle ausgegebenen Boxen pro Frame exportieren, einschließlich fehlender Personen-ID
   und leerer Frames. Tracking-ID und Personen-ID getrennt halten. Den tatsächlichen
   Entscheidungsframe unabhängig vom Snapshotframe speichern.
3. Versuchseinheiten mit eigenem Datenbank-Ausgangszustand, frischem Tracker und
   kollisionsfreien Snapshotpfaden ausführen. Für UC-12 Datenbank teilen, Tracker neu starten.
4. Vollständige Konfiguration, Eingabedatei-/Gewichte-Hashes, Codeversion, Hardware und
   Zeitmessungen in einem Laufmanifest sichern. `max_frames=0` für vollständige Videos nutzen.
5. Kleine Referenzfälle und einen annotierten Pilotclip durch Export und Auswertung führen.
   Danach Parameter und Codeversion für die Testclips festhalten.

Weiterhin bekannte technische Grenzen: Ersatz-IDs bei fehlender Tracker-ID,
kein expliziter Ablauf alter Track-Zustände, mögliche Profilverunreinigung nach ID-Switches,
Snapshot-Namen ohne Laufkennung und keine Encoder-Versionsprüfung bei gleicher Dimension.
Diese Punkte nicht als bereits behoben ausweisen. Profilupdate-Änderungen vor dem
Einfrieren der Methode entscheiden und in allen drei Varianten konsistent halten.

## Prüfungen für diesen Branch

`python -m unittest discover -s tests -v` prüft den ReID-Umfang und die konkreten
Korrekturen mit temporären Daten und Fake-Komponenten. Damit wird weder eine
ReID-Genauigkeit noch eine Echtzeitfähigkeit nachgewiesen. Das Paper enthält dafür
weiterhin den Versuchsplan und noch nicht erhobene Ergebnisse.
