# Vier-Video-Evaluation für G1 bis G4

Das Skript [`scripts/run_four_group_evaluation.py`](../scripts/run_four_group_evaluation.py)
verarbeitet je eine vorhandene Aufnahme aus G1 bis G4. Es verwendet das gespeicherte
Preset `best-calibrated` als B0 und leitet A2 und A3 zur Laufzeit exakt daraus ab.
Damit behalten alle Varianten unter anderem die kalibrierten Matchingwerte sowie
dieselben Zeit-, Überlappungs- und Modellparameter.

## Festgelegte Zuordnung

| Gruppe | ZIP-Datei | Szenario |
|---|---|---|
| G1 | `WhatsApp Video 2026-09-19 at 14.39.42.mp4` | Freie Sicht und unbekannter Eintritt |
| G2 | `WhatsApp Video 2026-09-19 at 14.39.49.mp4` | Verlassen und Rückkehr |
| G3 | `WhatsApp Video 2026-09-19 at 14.40.01.mp4` | Ähnliche schwarze Kleidung und Rückkehr |
| G4 | `WhatsApp Video 2026-09-19 at 14.39.55.mp4` | Kreuzung oder Verdeckung |

Die Zuordnung kann beim Aufruf mit `--group-video G2=anderer-name.mp4` geändert
werden. Jede Gruppe muss auf eine andere Datei im ZIP zeigen.

## Vorprüfung

```powershell
python scripts/run_four_group_evaluation.py --dry-run
```

Die Vorprüfung kontrolliert das ZIP, die Zuordnung und das Baseline-Preset. Sie
extrahiert keine Videos und startet keine Modelle.

## Vollständiger Lauf

```powershell
python scripts/run_four_group_evaluation.py --device auto
```

Es entstehen zwölf isolierte Läufe: vier Gruppen mal B0, A2 und A3. Jeder Lauf
beginnt mit einem leeren Profilbestand. Dadurch beeinflussen sich weder Gruppen
noch Varianten gegenseitig. Alle Videos werden vollständig verarbeitet.

Unter `data/experiments/four_groups_<lauf-id>/` liegen anschließend:

- `batch_manifest.json`: ZIP-Hash, Gruppenzuordnung und vollständige Parameter aller Varianten.
- `automatic_run_summary.csv`: technische Zähler, Laufzeit, FPS und Real-Time-Faktor.
- `automatic_paper_summary.md`: kompakte Tabelle der automatisch messbaren Werte.
- `manual_event_log.csv`: Vorlage für die Prüfung einzelner GT-Ereignisse.
- `paper_metrics_template.csv`: Vorlage für die aggregierten Paper-Kennzahlen.
- `evaluated_event_log.csv` und `evaluated_update_log.csv`: aus der geprüften Referenz erzeugte Ereignis- und Updateentscheidungen.
- `evaluation_results.json` und `paper_evaluation.md`: aggregierte Resultate für die Dokumentation.
- `G1/` bis `G4/`: vollständige Laufmanifeste, Frame-Exporte, Videos, Snapshots und Datenbanken.

Der Lauf vom 20. September 2026 wurde vollständig gesichtet und mit der geprüften
Ereignisreferenz ausgewertet. Die reproduzierbare Kurzfassung und alle Kennzahlen
stehen unter [Evaluationsresultate](evaluation/four_group_results.md). Für einen
neuen Lauf bleiben die automatisch erzeugten Identitäts- und Updatezähler zunächst
Systemausgaben; ihre reale Personenzuordnung muss wieder anhand der Videos und
Frame-Exporte geprüft werden.

Vier Videos ergeben eine kleine Fallstudie mit je einer Aufnahme pro Gruppe und
zwölf Verarbeitungsläufen. Dieser Bestand bildet den im Paper berichteten Umfang;
Versuchsaufbau, Stichprobengröße und Aussagegrenzen sind dort entsprechend
angepasst.
