# Einstieg in den ReID-Kern

Dieser Guide beschreibt den aktiven `main`-Stand. Den Versionsumfang und die noch
offenen Voraussetzungen für Messungen beschreibt [EVALUATION_SCOPE.md](EVALUATION_SCOPE.md).

## Empfohlene Lesereihenfolge

1. [config.py](../app/config.py): Pfade und Laufzeitparameter.
2. [models.py](../app/storage/models.py): Detection, MatchResult, PersonRecord und PipelineResult.
3. [default_mode.py](../app/modes/default_mode.py): B0 und die beiden gezielt abgeleiteten Varianten A1/A2.
4. [main.py](../app/main.py): CLI, Preset, Overrides und Pipeline-Aufruf.
5. [orchestrator.py](../app/pipeline/orchestrator.py): kompletter Ablauf.
6. [reid_encoder.py](../app/pipeline/reid_encoder.py) und [detector_tracker.py](../app/pipeline/detector_tracker.py): Modelladapter.
7. [vector_store.py](../app/storage/vector_store.py): Ähnlichkeitssuche und Profilupdates.
8. [streamlit_app.py](../app/ui/streamlit_app.py): interaktive Bedienung.

## Verantwortlichkeiten

| Modul | Aufgabe |
|---|---|
| `app/modes/` | ReID-Presets; alle verwenden denselben Hauptpfad |
| `app/pipeline/` | Tracking, Encoding und Ablaufsteuerung |
| `app/storage/` | persistente Profile, Ereignisse und Laufdaten |
| `app/utils/image_utils.py` | Crops, Qualität, Vektoren und Annotation |
| `app/utils/camera_utils.py` | lokale Kameraauswahl und Capture |
| `app/utils/upload_utils.py` | sichere, wiederverwendbare Upload-Dateien |
| `app/ui/` | Eingaben, Fortschritt und Ergebnistabellen |

## Ein Lauf

`PersonReIdPipeline` erzeugt im Konstruktor den Store. Tracker, Encoder und Store
können für Tests injiziert werden. `process()` erstellt zuerst Laufartefakte und
registriert den Lauf, lädt danach die Modelle, öffnet die Quelle und sichert die
Ressourcenfreigabe. Eine Pipelineinstanz darf nur eine Quelle verarbeiten.
`_process_open_capture()` liest Metadaten, öffnet den Writer, registriert den Lauf
und verarbeitet Frames.

Je Frame:

1. YOLO und Tracker liefern Boxen und Track-IDs.
2. Eine separate Ausgabekopie verhindert, dass Annotationen die Crops verändern.
3. Bereits sichtbare Personen-IDs werden für bekannte Tracks reserviert.
4. Unbekannte Tracks sammeln geeignete Crops; bekannte Tracks werden periodisch aktualisiert.
5. Ein neuer Track sucht mit dem kombinierten Embedding ein Profil oder erhält eine neue Personen-ID.
6. Profil und ausgewähltes Ereignis werden gespeichert.
7. Ausgabekopie und vollständigen Frame-Export schreiben, gegebenenfalls Vorschau aktualisieren.

## Zustände verstehen

| Zustand | Bedeutung | Gültigkeit |
|---|---|---|
| `track_to_person` | Zuordnung Tracker-ID zu Personenprofil | Aufruf von `process()` |
| `track_candidates` | akzeptierte Crops vor erster Zuweisung | Aufruf, Inhalt bis zur Zuordnung |
| `track_to_last_score` | letzter initialer Match-Score | Aufruf |
| `used_person_ids_this_frame` | aktuell reservierte Personenprofile | einzelner Frame |
| SQLite-Profil | normalisiertes, qualitätsgewichtet aktualisiertes Embedding | über Läufe hinweg |

Ein später angezeigter Score ist nicht automatisch eine neue Identitätsprüfung.
Der bestehende Track behält seine Zuordnung. Profilupdates nach Tracker-ID-Switches
und das Verwerfen veralteter Track-Zustände sind weiterhin offene fachliche Arbeiten.

## Presets und Konfiguration

`ModeConfig.to_pipeline_config()` überträgt den Namen nach `mode_name` und wendet
Overrides zuletzt an. B0 hat die ID `default`, A1 `colorhist` und A2
`no_quality_thresholds`. A1/A2 werden aus B0 abgeleitet, damit ihre Unterschiede
im Code überprüfbar bleiben.

Eigene Presets werden unter `AppPaths.mode_config_path` gespeichert:
`data/modes/reid_presets.json`. Es werden nur `person_reid`-Presets akzeptiert.
Built-in-IDs dürfen nicht durch gespeicherte Presets überschrieben werden.

Die UI verwendet nur einen Parametereditor. `app/ui/config_editor.py` definiert
die Laufzeitfelder aus `PipelineConfig`, vergleicht aktuelle Werte mit dem geladenen
Preset und überträgt dieselben Werte an Pipeline und neues Preset. Metadaten wie
Name und Beschreibung werden beim Speichern separat vergeben. Ein geänderter
Laufname erhält den Zusatz "(geändert)"; das Ausgangspreset bleibt unverändert.

`streamlit_app.py` hält Editorwerte in expliziten `pipeline_*`-Session-Schlüsseln.
Sie werden nur beim Presetwechsel oder beim expliziten Zurücksetzen neu geladen,
nicht bei jeder Interaktion. Nach dem Speichern wird die Auswahl vor der nächsten
Widget-Erzeugung auf das neue Preset gesetzt. Neue Laufzeitfelder müssen auch ein
Editorwidget erhalten; ein Test prüft die vollständige, eindeutige Feldabdeckung.
Alle effektiven `PipelineConfig`-Werte werden in `analysis_runs.metadata_json`
gespeichert. `app/evaluation/artifacts.py` ergänzt das Laufmanifest mit Hashes,
Hardwaredaten, Paketversionen, Zeitmessungen und Abschlussstatus.
`app/evaluation/runner.py` erzeugt frische Versuchsdatenbanken und neue Pipelines
pro Quelle; nur zusammengehörige Quellen innerhalb einer Einheit teilen Profile.

## Datenbank und Matching

`SQLiteVectorStore.search()` lädt alle Personenprofile, überspringt andere
Vektordimensionen und vergleicht mit Cosine Similarity. Die Suche kostet ungefähr
`O(Personenzahl x Dimension)`. Für den kleinen Versuchsbestand ist SQLite geeignet.

Gleiche Dimension garantiert keine kompatiblen Modellgewichte. Ein Encoderwechsel
braucht einen definierten neuen Bestand. Neue Datenbanken enthalten nur die drei
ReID-Tabellen. Alte Zusatzdaten werden nicht gelöscht.

Snapshots und Datenbank-Events sind Diagnosehilfen. `frames.jsonl` enthält dagegen
alle ausgegebenen Boxen und auch leere Frames. Fehlende Track-/Personenkennungen
bleiben null; der tatsächliche Entscheidungsframe wird nicht auf den früheren
Snapshotframe zurückdatiert. Details: [EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md).

## Häufige Fehler gezielt untersuchen

- Keine Profile: Detektionen, Mindestgröße, Qualitätswert und Anzahl akzeptierter Initialframes prüfen.
- Zu viele Profile: Track-Verlust, Match-Schwelle und Encoderbestand vergleichen.
- Vermischte Personen: initiale Matches und anschließende Profilupdates getrennt untersuchen.
- Fehlendes Video: Writer, Codec und Frameabmessungen prüfen.
- Fehler nach Modellwechsel: Dimension, Gewichte und Ausgangsdatenbank kontrollieren.

Parameter nur auf Pilotdaten abstimmen. Ein plausibles Ausgabevideo belegt keine
gemessene ReID-Qualität.

## Kleine Änderungen gut absichern

Zuerst einen Regressionstest für das beobachtete Verhalten ergänzen. Die Tests
verwenden temporäre Dateien und Fake-Komponenten; ein GPU-Lauf ist dafür nicht nötig.

Als nächste Ausbauschritte eignen sich injizierbare Modell-/Store-Abhängigkeiten,
ein separates Exportmodul und klar getrennte Matching-/Updatefunktionen.
Der wissenschaftliche Vergleich braucht zuerst verlässliche Messdaten.

Die ausgegliederten Erweiterungen und frühere Dokumentation sind auf
`codex/research-extensions` gesichert.
