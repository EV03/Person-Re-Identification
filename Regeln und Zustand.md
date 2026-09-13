# Projektstand - ReID-Masterprojekt

Der Branch `main` dokumentiert den ReID-Kern für die geplanten Versuche.
Verbindliche Einstiegspunkte sind [README.md](README.md),
[Codebase Guide](docs/CODEBASE_GUIDE.md) und
[Evaluationsstand](docs/EVALUATION_SCOPE.md).

## Aktueller Umfang

Video/Webcam, YOLO-Personendetektion, ByteTrack/BoT-SORT, qualitätsgefilterte
Person-Crops, Initialpuffer, OSNet/Farbhistogramm, Cosine Matching, synthetische
Personen-IDs, SQLite, annotierte Videos und Streamlit-Vorschau.

B0/A1/A2/A3 sind als Ausgangspresets vorhanden. OSNet-x1.0 nutzt dokumentierte MSMT17-Gewichte
ohne eigenes Fine-Tuning. Vollständige Frame-Exporte, getrennte Entscheidungs-/
Snapshotframes, isolierte Versuchseinheiten und Laufmanifeste sind implementiert.
Die quantitative Evaluation und Auswertung annotierter Clips stehen noch aus.
Es liegen keine aus diesem Stand erhobenen quantitativen Ergebnisse vor.
Der reduzierte Plan umfasst vier Versuchsgruppen und zwölf getrennte Testsequenzen,
jeweils mit vier eingefrorenen Pilotkonfigurationen (48 Kernläufe). Annotiert werden
Ereignisse und ausgewählte Übergangsfenster, keine vollständigen Trackingtrajektorien.

## Versionsabgrenzung

Der vorherige Umfang einschließlich Fußballanalyse-Platzhaltern, Bewegungsdiagnostik
und älteren Dokumentationsfassungen bleibt auf `codex/research-extensions` erhalten.
Diese Zusatzfunktionen gehören nicht zu den evaluierten Komponenten auf `main`.
Die Versuchsaufnahmen dürfen weiterhin Gehen, Kreuzung und Richtungswechsel enthalten.

Historische Datenbanken, Videoausgaben und lokale Presets bleiben erhalten.
Neue ReID-Presets verwenden `data/modes/reid_presets.json`; die bisherige
`custom_modes.json` wird nicht überschrieben.

## Änderungen dieses Stands

- Typisierte Tracker-, Matching-, Profilupdate- und Speicherverträge ergänzt.
- Matching und Profilupdate aus SQLite in austauschbare Policies/Service verschoben.
- Gemeinsame Konfigurationsfelder in `PipelineSettings` zusammengeführt.
- Profilupdate auf exakte gewichtete Summe aller akzeptierten Crops umgestellt.
- Ähnlichkeitsschutz vor Updates ergänzt; Ablehnungen verändern das Profil nicht.
- Standard-Einstiege trennen Datenbanken pro Encoder/Checkpoint und zusätzlich pro Versuchseinheit.

- Bestehende Fixes für Ressourcenfreigabe, Reset-Pfade und Upload-Lebenszyklus integriert.
- Zusatzmodule, ihre Konfigurationsfelder, UI-Elemente und Schema-Erzeugung entfernt.
- B0/A1/A2/A3 aus einer gemeinsamen Basis abgeleitet und Preset-Namen korrekt übertragen.
- Analysebild und Annotation getrennt.
- Sichtbare bekannte Personen-IDs vor dem Matching neuer Tracks reserviert.
- Dokumentation auf den tatsächlichen Hauptpfad und offene Evaluationsvoraussetzungen ausgerichtet.

## Arbeitsweise

App und Tests aus der Projektumgebung starten. Änderungen an Modell, Vorverarbeitung
oder Zuordnungsregeln vor den Testläufen festhalten. Gleiche Embedding-Dimension
allein reicht nicht für die Wiederverwendung eines Personenbestands.
Alte Aussagen über erfolgreiche Modellinitialisierung sind kein Nachweis für
ReID-trainierte Gewichte oder wissenschaftlich gemessene Erkennungsqualität.
