# TODO / Roadmap – Person Re-Identification

Stand: 25.08.2026

Dieses Dokument bündelt den aktuellen Projektstand, bekannte technische Schulden und mögliche Erweiterungen. Die Reihenfolge ist eine vorgeschlagene Priorisierung und noch keine Freigabe, Punkte ohne Rückfrage umzusetzen.

## Arbeitsregeln

- Vor Architektur-, Hardware-, Datenschutz- oder Produktannahmen Rückfrage halten.
- Erst eine reproduzierbare Baseline schaffen, danach Genauigkeit und neue Sensoren ausbauen.
- Neue Verfahren immer gegen denselben Benchmark vergleichen.
- Emotionen, Stimme, Gesichter, Gangbild und Funksignale als besonders sensible Daten behandeln.
- Unsichere KI-Ergebnisse mit Konfidenz ausgeben und nicht als sichere Tatsachen darstellen.

## 1. Reproduzierbare Entwicklungsumgebung herstellen – höchste Priorität

Aktueller Zustand:

- [x] Ursache geprüft: `.venv` verweist auf eine nicht mehr vorhandene Python-3.10.8-Installation.
- [x] Reiner Syntaxcheck erfolgreich: 43 Python-Dateien sind syntaktisch gültig.
- [x] Zielversion festgelegt: projektgebundenes CPython 3.10.8 unter `.python/`.
- [x] Zielhardware festgelegt: Windows mit NVIDIA/CUDA und CPU-Fallback.
- [x] Torchreid/OSNet als verbindlichen ReID-Standard festgelegt; Color-Histogram-Encoder entfernt.
- [x] Defekte `.venv` kontrolliert ersetzt.
- [x] Basisabhängigkeiten installiert.
- [x] Torch 2.11.0+cu128, Torchvision 0.26.0+cu128, Torchreid 1.4.0 und OSNet geprüft.
- [x] Abhängigkeiten in `pyproject.toml` festgelegt und mit `uv.lock` reproduzierbar gesperrt.
- [x] `requirements.txt` an den gelockten Standard angeglichen; unterstützter Installationsweg ist uv.
- [x] Import-Smoke-Test für UI, Pipeline, SQLite, Qdrant, YOLO und Torchreid ausgeführt.
- [x] Vollständigen isolierten 528-Frame-Videotest ohne Veränderung der produktiven Datenbank ausgeführt.
- [x] CPU-Kurztest sowie CUDA-Test auf NVIDIA GeForce RTX 4070 ausgeführt.
- [x] GPU, CUDA, Modellpfade und 512D-Embedding protokolliert.
- [x] Setup-, Start-, Modellinstallations- und Smoke-Test-Skripte für Windows erstellt.
- [x] Setup-Anleitung und Supportstatus in `README.md` aktualisiert.
- [x] Lokale YOLO-/OSNet-Modellerkennung und getrennte UI-Auswahlfelder ergänzt.

Definition of Done:

- Ein frischer Checkout lässt sich mit dokumentierten Befehlen installieren und starten.
- Default-Modus, CLI und Streamlit-UI bestehen einen Smoke-Test.
- Die verwendeten Paketversionen und das CPU-/CUDA-Verhalten sind reproduzierbar.

Status: lokal vollständig umgesetzt und verifiziert. Linux ist nicht unterstützt; AMD/ROCm ist nur als ungetestete Erweiterungsmöglichkeit dokumentiert.

## 2. Automatisierte Tests und verlässliche Evaluation

- [ ] `pytest`-Teststruktur einführen.
- [ ] Unit-Tests für Matching-Zonen, Quality Gate, Embedding-Dimensionen und Same-Frame-Ausschluss ergänzen.
- [ ] SQLite- und Qdrant-Store-Vertrag mit denselben Tests prüfen.
- [ ] Tests für Manifest-Parsing, Metriken und Modus-Konfiguration ergänzen.
- [x] Reproduzierbaren, isolierten Video-Integrationstest für Windows bereitstellen.
- [ ] CI-Workflow für Syntax, Linting und Tests einrichten.
- [ ] Adaptive Evaluation nur starten, wenn genau ein Kalibrierungsvideo vorhanden ist.
- [ ] Ungültige Manifeste mit klarer Fehlermeldung ablehnen.
- [ ] Abgebrochene Evaluationsläufe als `failed` oder `cancelled` markieren.
- [ ] Den unvollständigen Lauf `eval_20260702_232235` untersuchen und dokumentieren.
- [ ] Auf dem aktuellen Commit Vergleichsläufe für `fixed_db`, `learn_through` und `adaptive_calibration` durchführen.
- [ ] Ergebnisse nach Bedingung, Kamera, Kleidung, Pose und Bewegungsart vergleichen.
- [ ] Multi-Person-, ähnlich gekleidete Personen- und echte Unbekannt-Personen-Tests ergänzen.
- [ ] Cross-Camera-Testdaten und Verdeckungs-/Wiedereintrittsszenarien ergänzen.
- [ ] Precision, Recall, False Match Rate, False Non-Match Rate, IDF1/HOTA und Konfidenzverteilung ergänzen.

## 3. Detection, Tracking und Bewegungsstabilität verbessern

- [ ] ByteTrack und BoT-SORT unter identischen Bedingungen benchmarken.
- [ ] Eigenes Tracker-YAML für Track Buffer, Match Threshold und Verdeckungen testen.
- [ ] YOLOv8n, YOLOv8s und gegebenenfalls neuere kompatible Detektoren vergleichen.
- [ ] Bildgrößen und Detection Confidence systematisch kalibrieren.
- [ ] Kamerabewegungsausgleich/GMC prüfen.
- [ ] ID-Switches, Track-Sprünge und verlorene Tracks separat messen.
- [ ] Vorhandene Bewegungsrichtung mit falschen Matches und ID-Switches korrelieren.
- [ ] Motion-Bonus erst nach Benchmark-Ergebnissen anpassen oder als harte Regel aktivieren.
- [ ] Verdeckungsbehandlung und Wiederaufnahme eines verlorenen Tracks verbessern.
- [ ] Optional Segmentierung statt reiner Bounding Boxes evaluieren.

## 4. Personen- und Profilverwaltung

- [ ] UI für Personenliste, bestes Snapshot-Bild und Profilhistorie ausbauen.
- [ ] Unsichere/Pending-Matches zur manuellen Prüfung anzeigen.
- [ ] Zwei Profile kontrolliert zusammenführen.
- [ ] Falsch zusammengeführte Profile wieder trennen.
- [ ] Einzelne falsche Samples oder Events entfernen.
- [ ] Person vollständig löschen.
- [ ] Datenbank-Reset mit Warnung und Bestätigung im UI anbieten.
- [ ] Merge-Vorschläge aus vorhandenen Merge-Grenzwerten erzeugen; keine automatische Verschmelzung ohne Tests.
- [ ] Änderungen an Profilen revisionssicher protokollieren.
- [ ] Anzahl ähnlicher Embedding-Samples begrenzen und repräsentative Samples auswählen.
- [ ] Alte, redundante oder qualitativ schlechte Samples bereinigen.
- [ ] Embedding-Modell und Embedding-Version pro Profil speichern.

## 5. Datenhaltung, Qdrant und Repository-Hygiene

- [ ] Bereits eingecheckte lokale Qdrant-Laufzeitdaten aus der Versionsverwaltung entfernen.
- [ ] Umgang mit der unversionierten `app.zip` festlegen: löschen, ignorieren oder bewusst archivieren.
- [ ] Datenbankschema versionieren und Migrationen ergänzen.
- [ ] Inkompatible Embedding-Dimensionen automatisch erkennen und sicher behandeln.
- [ ] Gleichzeitige Streamlit-/Qdrant-Zugriffe testen und sperren oder koordinieren.
- [ ] Backup-, Restore- und Exportfunktion ergänzen.
- [ ] Löschfristen für Events, Videos, Snapshots und Embeddings konfigurierbar machen.
- [ ] PostgreSQL nur bei echtem Mehrbenutzer-/Metadatenbedarf evaluieren.
- [ ] README, tatsächliche Defaults und Qdrant-Status miteinander synchronisieren.

## 6. UI, Diagnose und Betrieb

- [ ] Sichtbaren Systemstatus ergänzen: Detector, Tracker, Encoder, Embedding-Dimension, Store und Device.
- [ ] CUDA-Verfügbarkeit und GPU-Name anzeigen.
- [ ] Verständliche Installations-, Modell- und Datenbankfehler anzeigen.
- [ ] Laufstatus `queued/running/completed/failed/cancelled` speichern.
- [ ] Abbruchknopf für lange Video- und Evaluationsläufe ergänzen.
- [ ] Ergebnisberichte als CSV/JSON/PDF exportierbar machen.
- [ ] Remote-Browserkamera mit WebRTC als Alternative zur lokalen OpenCV-Webcam evaluieren.
- [ ] Rollen, Anmeldung und Zugriffsrechte vor einem Mehrbenutzerbetrieb ergänzen.

## 7. Mehrere Kameras und kamerübergreifende Re-Identification

- [ ] Ziel klären: Live-Kameras, aufgezeichnete Videos oder beides.
- [ ] Kamera-Registry mit stabiler `camera_id`, Standort und Metadaten einführen.
- [ ] Zeitstempel synchronisieren; Drift und unterschiedliche FPS berücksichtigen.
- [ ] Pro Kamera lokale Track-IDs und global getrennte Personen-IDs modellieren.
- [ ] Cross-Camera-ReID mit Kamera- und Zeitfenstern implementieren.
- [ ] Zulässige Übergänge und Reisezeiten zwischen Kamerazonen modellieren.
- [ ] Gleichzeitige Sichtbarkeit derselben globalen ID auf unvereinbaren Kameras verhindern.
- [ ] Kalibrierung von überlappenden Kameras und gemeinsame Bodenebene prüfen.
- [ ] Ereignisbus/Queue für mehrere Kamera-Worker evaluieren.
- [ ] Lasttests für parallele Streams, GPU-Speicher und Backpressure durchführen.
- [ ] Multi-Camera-Dashboard mit Kameraansicht, Übergängen und globaler Identität bauen.

## 8. Pose-, Skelett- und Aktionserkennung als optionaler Nebenservice

Zielidee: Auf erkannte Personen ein Pose-Skelett legen und daraus Bewegungen/Aktionen schätzen, ohne die stabile ReID-Pipeline direkt zu blockieren.

- [ ] Gewünschte Aktionen und Anwendungsfälle festlegen, bevor ein Modell gewählt wird.
- [ ] MediaPipe Pose, YOLO Pose und weitere geeignete Pose-Modelle benchmarken.
- [ ] Optionalen Pose-Service mit klarer Eingabe/Ausgabe definieren.
- [ ] Person-Crop und `track_id` an den Service übergeben.
- [ ] Keypoints, Sichtbarkeit, Konfidenz und Zeitstempel zurückgeben.
- [ ] Skelett im Live- und Ergebnisvideo einblendbar machen.
- [ ] Temporale Glättung für flackernde oder verdeckte Keypoints ergänzen.
- [ ] Einfache Aktionen wie stehen, gehen, laufen, sitzen, hinfallen, winken oder Arm heben evaluieren.
- [ ] Action Recognition über mehrere Frames statt nur über eine Einzelpose aufbauen.
- [ ] Gangbild/Gait nur als gesondertes sensibles Forschungsmodul und mit Einwilligung prüfen.
- [ ] Laufzeitbudget definieren, damit Pose-Erkennung Detection/ReID nicht ausbremst.
- [ ] Prüfen, ob ein Prozess, lokaler HTTP/gRPC-Service oder eine gemeinsame GPU-Pipeline sinnvoller ist.

## 9. Sprechaktivität und Emotionserkennung – Forschungsbereich

### Sprechaktivität

- [ ] Klären, ob nur „Person spricht gerade“ oder auch Sprecheridentität/Transkription benötigt wird.
- [ ] Visuelle Sprechaktivität über Mund-/Lippenbewegung evaluieren.
- [ ] Optional Audio-VAD zur Erkennung von Sprachabschnitten ergänzen.
- [ ] Optional Active-Speaker-Detection aus Audio und Video kombinieren.
- [ ] Audio- und Videotimestamps synchronisieren.
- [ ] Ergebnis als Wahrscheinlichkeit mit Quelle `visual/audio/fused` speichern.
- [ ] Keine Gesprächsinhalte speichern oder transkribieren, solange dies nicht ausdrücklich beschlossen ist.

### Emotionen

- [ ] Konkreten, zulässigen Anwendungsfall und Einwilligung klären.
- [ ] Emotionserkennung nicht als zuverlässige Aussage über den inneren Zustand einer Person darstellen.
- [ ] Falls erforscht: nur sichtbare Ausdrucksmerkmale mit Unsicherheit klassifizieren.
- [ ] Verzerrungen nach Licht, Blickwinkel, Kultur, Hautfarbe, Alter und Verdeckung evaluieren.
- [ ] Keine automatischen Hochrisikoentscheidungen auf Emotionsschätzungen stützen.
- [ ] Speicherung standardmäßig deaktivieren und kurze Löschfristen verwenden.

## 10. WLAN-/Wi-Fi-Sensing als Ergänzung zu Computer Vision

Zielidee: Funksignale als zusätzliches Bewegungs-/Anwesenheitssignal verwenden, nicht automatisch als Identitätsbeweis.

- [ ] Anwendungsfall klären: Anwesenheit, Bewegung, Raumwechsel, Atmung/Gesten oder Identität.
- [ ] Verfügbare Hardware und erlaubte Messdaten prüfen.
- [ ] Gewöhnliche RSSI-Werte von echtem Wi-Fi-CSI unterscheiden.
- [ ] Prüfen, ob Netzwerkkarte, Firmware und Treiber CSI-Daten liefern können.
- [ ] Alternativ UWB oder mmWave-Radar vergleichen, wenn CSI-Hardware ungeeignet ist.
- [ ] Separaten Sensor-Service für Zeitstempel, Sensor-ID, Signalmerkmale und Konfidenz entwerfen.
- [ ] Zeitsynchronisation mit Kameraereignissen implementieren.
- [ ] Fusion zunächst nur für Anwesenheits- und Bewegungsplausibilität testen.
- [ ] Controlled-Room-Prototyp mit und ohne Sichtkontakt durchführen.
- [ ] Fehlalarme durch Wände, andere Personen, Geräte und wechselnde Raumgeometrie messen.
- [ ] Datenschutz, Funkrecht, Einwilligung und Missbrauchsrisiken vor Feldtests prüfen.

## 11. Weitere ergänzende Sensortechnologien

- [ ] RGB-D-/Tiefenkamera für Abstand, 3D-Pose und bessere Verdeckungsbehandlung evaluieren.
- [ ] Wärmebildkamera für Anwesenheit bei schlechtem Licht evaluieren.
- [ ] UWB-Tags für explizit kooperative, hochgenaue Positionsbestimmung evaluieren.
- [ ] BLE-Beacons nur für grobe Nähe-/Raumerkennung prüfen.
- [ ] mmWave-Radar für Bewegung und Anwesenheit ohne sichtbares RGB-Bild prüfen.
- [ ] Sensorfusion modular halten; jedes Signal behält Quelle, Zeitstempel und Unsicherheit.

## 12. Football-Team-Analysis fertigstellen

- [ ] Team-Farbklassifikation in die Crop-Verarbeitung integrieren.
- [ ] Spieler, Torwart und Schiedsrichter voneinander unterscheiden.
- [ ] Echten Ball-Detektor statt `BallDetectorPlaceholder` integrieren.
- [ ] Pitch-Keypoints oder manuelle Spielfeldkalibrierung ergänzen.
- [ ] Homographie und Bounding-Box-Fußpunkt in Meterkoordinaten umrechnen.
- [ ] `player_frame_events` und `ball_frame_events` befüllen.
- [ ] Distanz, Geschwindigkeit, Sprint, Heatmap und Ballnähe aggregieren.
- [ ] Football-Orchestrator implementieren und an die Mode Registry anschließen.
- [ ] Football-Dashboard und Exporte erstellen.
- [ ] Ergebnisse gegen annotierte Spielszenen validieren.

## 13. Detailerkennung weiterentwickeln

- [ ] Nutzen der aktuellen OpenCV-Heuristiken quantitativ messen.
- [ ] Fehlerhafte oder volatile Attribute schwächer gewichten oder deaktivieren.
- [ ] Trainierten Attribut-Classifier nur bei nachgewiesenem Mehrwert evaluieren.
- [ ] Körperbereiche posebasiert in Kopf, Torso und Unterkörper aufteilen.
- [ ] Segmentierte Körper-Crops gegen rechteckige Crops benchmarken.
- [ ] Detailmerkmale weiterhin nur als unterstützendes Signal für OSNet verwenden.

## 14. Datenschutz, Sicherheit und Governance

- [ ] Rechtsgrundlage und Einwilligung je Anwendungsfall dokumentieren.
- [ ] Datenminimierung und Zweckbindung technisch umsetzen.
- [ ] Rollen- und Rechtekonzept erstellen.
- [ ] Audit-Logging für Anzeige, Export, Merge und Löschung ergänzen.
- [ ] Verschlüsselung für sensible Daten und Backups prüfen.
- [ ] Lösch- und Aufbewahrungsfristen umsetzen.
- [ ] Modellkarten für ReID, Pose, Sprechaktivität und Emotionen erstellen.
- [ ] Keine echten Namen standardmäßig an synthetische IDs binden.
- [ ] Gesicht, Stimme, Gangbild, Emotion und Funksensing nur als separat freizugebende Module behandeln.

## Vorgeschlagene Reihenfolge

1. Reproduzierbare Entwicklungsumgebung
2. Tests und aktuelle Baseline-Evaluation
3. Tracking- und ReID-Stabilität
4. Personenverwaltung und Datenlebenszyklus
5. Multi-Camera-Grundarchitektur
6. Pose-/Action-Nebenservice
7. Football-Modus oder Sprechaktivität – abhängig vom Produktziel
8. Wi-Fi-/Radar-Sensorfusion als gesonderter Forschungsprototyp
9. Emotionserkennung nur nach klarer fachlicher und rechtlicher Freigabe
