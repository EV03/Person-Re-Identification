# Evaluationsumfang auf main

Stand: 13. September 2026. Dieser Branch enthält den abgegrenzten ReID-Prototyp.
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
| `no_update_similarity` | A3 | Nur der Update-Ähnlichkeitsschutz wird mit -1 deaktiviert |
| `details_tracking` | D1 | Detail-Re-Ranking, Strong/Weak/Low-Zonen, Evidenz vor neuen IDs und begrenzter räumlicher Bonus |

A2 behält Qualitätsgewichtung, Mindestgrößen, Initialpuffer und Snapshot-Auswahl.
B0/A1/A2/A3 bleiben der vierteilige Kernvergleich; D1 ist eine getrennt
ausgewiesene Erweiterungspipeline. Alle fünf Presets sind Ausgangskonfigurationen,
keine final ausgewählten Pilotwerte.
Sie enthalten zunächst dieselben sonstigen Parameter. CLI/UI-Overrides
sind möglich und müssen als Konfigurationsänderung aufgezeichnet werden.

Die UI lädt Presets in einen gemeinsamen, vollständigen Pipeline-Editor. Starten
und "Aktuelle Einstellungen speichern" verwenden dieselben Parameter. Geänderte
Läufe werden im Namen markiert; die effektive Pipeline-Konfiguration ist einsehbar
und wird vollständig in `analysis_runs.metadata_json` festgehalten. Das Speichern
erstellt ein neues Preset und lädt es; vorhandene Presets werden nicht überschrieben.
Alle fünf konfigurierbaren Matching-/Konfidenz-/Qualitätsschwellen, Mindestgrößen
und zeitlichen Parameter sind editierbar. Tracker-interne Schwellen bleiben in der
Tracker-YAML, Konstanten und Gewichte der Qualitätsheuristik bleiben unverändert.
Diese Einstellbarkeit dient Pilotversuchen, nicht einer Nachkalibrierung auf Testclips.

Die bisherige umfangreiche Bewegungsdiagnostik bleibt aus der normalen `main`-
Policy entfernt. Die über das D1-Preset wählbare `Details_Tracking`-Policy kann ausschließlich
einen kleinen, begrenzten Bildraum-Kontinuitätsbonus verwenden. Richtung,
Pixelgeschwindigkeit und Sprungmetriken werden weiterhin nicht als eigene
Identitätsmethode behauptet. Die interne Bewegungsschätzung des verwendeten
Trackers bleibt davon getrennt.

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

Die normale `main`-Policy behält weiterhin ihre Personenkennung für bekannte
Tracks. Für getrennt ausgewiesene Erweiterungsläufe steht nun das D1-Preset mit
der `Details_Tracking`-Policy, Detail-Re-Ranking, drei Entscheidungszonen,
verzögerter Neuanlage und kleinem räumlichen Bonus bereit. Profilupdates prüfen
weiterhin die Ähnlichkeit zum Zielprofil; Weak-Matches aktualisieren das Profil
nicht. A3 verändert nur den ursprünglichen Update-Schutz der `main`-Methode.

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

Tracker, Matching, Profilupdate und Ablage sind über typisierte Verträge
getrennt. Gemeinsame Settings stehen
nur in `PipelineSettings`. Erweiterungen: [EXTENDING_BACKENDS.md](EXTENDING_BACKENDS.md).
`WeightedMeanProfileUpdater` akkumuliert rohe qualitätsgewichtete Summen,
einschließlich aller Initial-Crops. Vor Updates schützt eine konfigurierbare
Ähnlichkeitsschwelle (Default 0,82) das bestehende Profil. Abgelehnte Versuche
werden protokolliert. Standard-Einstiege verwenden getrennte Datenbanken pro
Encoder-Konfiguration; isolierte Versuchseinheiten bleiben zusätzlich getrennt.
Alte DBs werden nicht übernommen. Formeln: [PROFILE_UPDATES.md](PROFILE_UPDATES.md).

## Reduzierter finaler Versuchsplan

Separate Pilotaufnahmen dienen der schrittweisen Wahl von Matching-, Update-
Ähnlichkeits- und Qualitätsschwellen. Danach vier eigene Presets speichern und vor
der Testauswertung einfrieren. A1 verwendet eigene Encoder-Ähnlichkeitsschwellen;
A2/A3 werden aus der kalibrierten B0 abgeleitet. Eingebaute Presets übernehmen
die bearbeiteten B0-Werte nicht automatisch.

Vier Versuchsgruppen: freie Sicht/unbekannter Eintritt, Rückkehr, ähnliche Kleidung
mit Rückkehr und Kreuzung/Verdeckung. Je drei getrennte Aufnahmen ergeben zwölf
Testsequenzen, jeweils vier Varianten und damit 48 Kernläufe. Ein Videopaar mit
gemeinsamen Profilen zählt als eine Sequenz. Externer Clip und Leerraum-Negativtest
sind optional; zusätzliche technische Laufzeitwiederholungen nur auf einer vorab
gewählten repräsentativen Sequenz je Variante, nicht auf allen zwölf.

GT-Annotation für Registrierung, Ein-/Austritt, Rückkehr und je G4-Aufnahme ein
dreisekündiges Übergangsfenster. Rückkehrentscheidungen, die Ausgabe nach zwei
Sekunden und Updateversuche in diesen Fenstern manuell anhand der Aufnahme und
Exporte prüfen. Fehlregistrierungen, fehlende IDs und Detektions-/Trackingfehler
bleiben im Ergebnis; nur GT-basierte Sichtbarkeit begründet vorab einen Ausschluss.
Keine vollständigen Precision-/Recall-, IDF1- oder ID-Switch-Zahlen ohne dichte
Referenztrajektorien. Der Versuchsstarter berechnet die GT-Metriken nicht automatisch;
für diese manuelle Ereignisevaluation ist TrackEval keine Pflicht.
Bedienung, Pilotrastersuche und Ergebnisprotokoll: [EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md).

Weiterhin bekannte Grenzen: kein expliziter Ablauf alter Track-Zustände,
mögliche Profilverunreinigung nach ID-Switches trotz Ähnlichkeitsschutz, keine
automatische Reparatur falscher Track-/Personenzuordnungen und nominale
FPS-Zeitstempel statt ursprünglicher VFR-PTS. Die Standardpfade trennen
Encoder-Konfigurationen; bei bewusst injizierten eigenen Speichern bleibt die
Trennung Verantwortung des Aufrufers. Änderungen nach dem Einfrieren getrennt
ausweisen. Vier Personen und Ereignisfenster erlauben keine breite Generalisierung.

## Prüfungen für diesen Branch

`python -m unittest discover -s tests -v` prüft den ReID-Umfang und die konkreten
Korrekturen mit temporären Daten und Fake-Komponenten. Zusätzlich laufen opt-in
echte Modell-/Video-Tests für alle eingebauten Varianten mit künstlichen Clips ohne Personen.
Beide echten Tests sind lokal erfolgreich. Damit wird weder eine
ReID-Genauigkeit noch eine Echtzeitfähigkeit nachgewiesen. Das Paper enthält dafür
weiterhin den reduzierten Versuchsplan und noch nicht erhobene Ergebnisse.
