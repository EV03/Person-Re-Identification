# Projektstand - ReID-Masterprojekt

Der Branch `main` dokumentiert den ReID-Kern für die geplanten Versuche.
Verbindliche Einstiegspunkte sind [README.md](README.md),
[Codebase Guide](docs/CODEBASE_GUIDE.md) und
[Evaluationsstand](docs/EVALUATION_SCOPE.md).

## Aktueller Umfang

Video/Webcam, YOLO-Personendetektion, ByteTrack/BoT-SORT, qualitätsgefilterte
Person-Crops, Initialpuffer, OSNet/Farbhistogramm, Cosine Matching, synthetische
Personen-IDs, SQLite, annotierte Videos und Streamlit-Vorschau.

B0/A1/A2 sind als Presets vorhanden. Ein vollständiger Vorhersageexport,
definierte Modellgewichte und automatisierte isolierte Versuchsläufe stehen noch aus.
Es liegen keine aus diesem Stand erhobenen quantitativen Ergebnisse vor.

## Versionsabgrenzung

Der vorherige Umfang einschließlich Fußballanalyse-Platzhaltern, Bewegungsdiagnostik
und älteren Dokumentationsfassungen bleibt auf `codex/research-extensions` erhalten.
Diese Zusatzfunktionen gehören nicht zu den evaluierten Komponenten auf `main`.
Die Versuchsaufnahmen dürfen weiterhin Gehen, Kreuzung und Richtungswechsel enthalten.

Historische Datenbanken, Videoausgaben und lokale Presets bleiben erhalten.
Neue ReID-Presets verwenden `data/modes/reid_presets.json`; die bisherige
`custom_modes.json` wird nicht überschrieben.

## Änderungen dieses Stands

- Bestehende Fixes für Ressourcenfreigabe, Reset-Pfade und Upload-Lebenszyklus integriert.
- Zusatzmodule, ihre Konfigurationsfelder, UI-Elemente und Schema-Erzeugung entfernt.
- B0/A1/A2 aus einer gemeinsamen Basis abgeleitet und Preset-Namen korrekt übertragen.
- Analysebild und Annotation getrennt.
- Sichtbare bekannte Personen-IDs vor dem Matching neuer Tracks reserviert.
- Dokumentation auf den tatsächlichen Hauptpfad und offene Evaluationsvoraussetzungen ausgerichtet.

## Arbeitsweise

App und Tests aus der Projektumgebung starten. Änderungen an Modell, Vorverarbeitung
oder Zuordnungsregeln vor den Testläufen festhalten. Gleiche Embedding-Dimension
allein reicht nicht für die Wiederverwendung eines Personenbestands.
Alte Aussagen über erfolgreiche Modellinitialisierung sind kein Nachweis für
ReID-trainierte Gewichte oder wissenschaftlich gemessene Erkennungsqualität.
