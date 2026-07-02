# Kalibrierungsmodus und Frontend-Überarbeitung

## Ziel

Die normale Streamlit-App unterstützt jetzt Kalibrierung nicht nur als Evaluation-Skript, sondern als regulären Betriebsmodus der Pipeline. Kalibrierung wird als initialer Profilaufbau verstanden. Das Profil darf danach kontrolliert wachsen.

## Neuer Kalibrierungsmodus in der App

Im Sidebar-Bereich **Kalibrierung im normalen Projekt** gibt es drei Modi:

| Modus | Bedeutung |
|---|---|
| Aus | normale ReID-Verarbeitung |
| Neue Person kalibrieren | erzeugt ein neues Personenprofil aus hochwertigen Crops |
| Bestehende Person erweitern | ergänzt ein vorhandenes Profil mit neuen hochwertigen Embeddings |

Für Kalibrierung werden automatisch strengere Qualitätswerte genutzt:

- höhere Inferenzauflösung,
- größere Mindest-Crop-Größe,
- höhere Mindestqualität für Embeddings,
- höhere Mindestqualität für Profilupdates,
- Detail-Gewicht standardmäßig deaktiviert beziehungsweise stark reduziert.

## Pipeline-Logik

Die Pipeline erhält neue Konfigurationsfelder:

```text
calibration_mode = off | new_person | extend_person
calibration_target_person_id
calibration_label
```

Bei `new_person` wird die erste ausreichend gute Kandidatenmenge als neues Personenprofil gespeichert. Weitere hochwertige Crops aus demselben Lauf erweitern dieses Profil.

Bei `extend_person` wird eine vorhandene `person_id` ausgewählt und mit neuen Embedding-Samples erweitert.

## Frontend-Verbesserungen

- Sidebar optisch überarbeitet.
- Parameter sind stärker gruppiert.
- Kalibrierung hat einen eigenen erklärten Bereich.
- Tabellen für Analysis Runs, Persons und Events sind einklappbar.
- Tabellen besitzen Spaltenauswahl und zusätzliche Erklärungen.
- Evaluation-Ergebnisse aus `all_videos_summary.csv` werden im Default-Modus tabellarisch angezeigt.
- Evaluation kann aus der App heraus gestartet werden.

## Evaluation-Ergebnisse

Im Default-Modus liest die App automatisch aus:

```text
data/evaluation_runs/<run_id>/all_videos_summary.csv
```

Die Tabelle kann gefiltert, durchsucht und in ihren Spalten reduziert werden.

## Hinweis

Lange Evaluation-Läufe können in Streamlit blockierend wirken. Für robuste Langläufe bleibt PowerShell weiterhin sinnvoll. Der UI-Button ist vor allem für kurze bis mittlere Tests gedacht.
