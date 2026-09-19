# Integration von `Details_Tracking` in `main`

Stand: 19. September 2026

## Warum kein normaler Merge möglich war

`main` und `Details_Tracking` besitzen keinen gemeinsamen Git-Merge-Base und
entwickelten unterschiedliche Architekturen. `main` trennt Tracker, Encoder,
Profilservice, Persistenz und Auswertungsartefakte über typisierte Verträge.
`Details_Tracking` bündelte mehr Entscheidungslogik direkt in Orchestrator und
Speicher. Die Funktionen wurden daher gezielt in die aktuelle Architektur
übertragen; alte Dateien wurden nicht über `main` kopiert.

## Übernommene ReID- und Trackingbestandteile

| Bestandteil | Aufgabe | Integration |
|---|---|---|
| Detail Registry | Schwache, erklärbare Merkmale wie Brille, Kappe, Ober-/Unterkörperfarbe und Textur aus Person-Crops gewinnen | `app/utils/detail_utils.py`; optionales Re-Ranking, niemals alleinige Identitätsentscheidung |
| Detail-Re-Ranking | OSNet-Cosine-Score leicht anheben oder absenken; `0,5` bleibt neutral | `final = visual + (detail - 0.5) * weight`, begrenzt auf `[-1, 1]` |
| Strong/Weak/Low-Zonen | Sichere Matches von unsicheren Kandidaten und klar niedrigen Scores trennen | D1-`DetailTrackingPolicy`; Weak erhält eine vorläufige ID, aktualisiert aber das Profil nicht |
| Verzögerte neue IDs | Eine neue Person erst nach wiederholten niedrigen Scores und ausreichender Zeitspanne anlegen | Laufbezogener Evidenzpuffer pro Track |
| Überlappungsschutz | Bei stark überlappenden Boxen im selben Frame keine vorschnelle zweite Identität erzeugen | Intersection-over-smaller-box gegen bereits zugewiesene Boxen |
| Räumlicher Kontinuitätsbonus | Kürzlich nahe beobachtete Person als kleinen Zusatzhinweis nutzen | Maximaler Bonus `0,04`; kein biometrisches Merkmal und kein Ersatz für den Tracker |
| Detail-Profil | Detailvektor qualitätsgewichtet zusammen mit dem Personenprofil fortschreiben | Optionale SQLite-Spalten; bestehende Datenbanken werden automatisch erweitert |
| Einzelpersonen-Test | Track-/Person-Kontinuität und Fragmentierung in kontrollierten Ein-Person-Videos messen | Importierbares Modul plus `scripts/evaluate_single_person_videos.py` |

Die normale `main`-Policy bleibt verfügbar. Dadurch ändern sich B0/A1/A2/A3
nicht stillschweigend. Die portierte Methodik ist als eigenes Preset
`details_tracking` / **D1 - Details Tracking** oben in der Streamlit-Seitenleiste
auswählbar. Entscheidungspolicy und sämtliche Detail-, Evidenz- und
Bewegungsparameter gehören zur gespeicherten Pipeline-Konfiguration. Die
tatsächlich verwendete Policy wird in das Laufmanifest geschrieben.

## D1-D6: Ablationen der Details-Pipeline

| Preset | Aktive Änderung gegenüber D1 |
|---|---|
| D1 `details_tracking` | Vollständige Details-Tracking-Pipeline |
| D2 `details_no_reranking` | Detail-Registry und Detail-Re-Ranking ausgeschaltet |
| D3 `details_no_weak_zone` | Keine Weak-Zone; unter Strong folgt direkt Low |
| D4 `details_immediate_new_person` | Keine Low-Evidenzsammlung; Neuanlage nach dem normalen Initialpuffer |
| D5 `details_no_overlap_protection` | Kein Überlappungsschutz vor der Neuanlage |
| D6 `details_no_motion_bonus` | Kein räumlicher Kontinuitätsbonus |

Alle übrigen Parameter bleiben gegenüber D1 identisch. Dadurch misst jeder
Vergleich den Einfluss genau eines entfernten Bestandteils. Die fünf Schalter
sind auch im vollständigen Pipeline-Editor sichtbar und werden in eigenen
Presets und Laufmanifesten gespeichert.

YOLO-Gewichte in `models/yolo`, `data/models` oder im Projektstamm sowie
ReID-Checkpoints in `models/reid`, `data/models` oder dem Torch-Checkpoint-Cache
werden automatisch in den Modell-Dropdowns angeboten. Eigene Pfade bleiben
eingebbar. Das ist eine lokale Dateierkennung, kein automatischer Download.

## Zwei Testmodule in der Web-UI

### Mehrpersonen- und Trackingtest

- geeignet für zwei oder mehr sichtbare Personen;
- zeigt die vorhandenen `main`-Artefakte (`frames.jsonl`, MOT-Export,
  Laufmanifest), Laufzeit, FPS und Real-Time-Factor;
- läuft wahlweise mit B0/A1/A2/A3, D1-D6 oder einem eigenen gespeicherten Preset;
- erzeugt ohne dichte Ground Truth bewusst keine behaupteten IDF1-/MOTA-Werte.

### Einzelpersonen- und Detailtest

- erwartet genau eine reale Person im Video;
- wertet die oben ausgewählte Pipeline aus, ohne deren Policy zu verändern;
- berechnet dominante Track-/Person-Quote, Track-/Person-Wechsel,
  Expected-Person-Quote, Profilwachstum und Fragmentierungsindex;
- speichert JSON-, Markdown- und CSV-Bericht beim Laufartefakt;
- ist zusätzlich per Manifest als `fixed_db`, `learn_through` oder
  `adaptive_calibration` ausführbar.

Die Auswahl des Testmoduls ändert ausschließlich die Auswertung: Beide
Testmodule können mit jeder Pipeline ausgeführt werden. Für einen
Details-Tracking-Lauf wird D1 oben als ReID-Preset gewählt; für denselben
Einzelpersonentest mit der Main-Methode wird beispielsweise B0 gewählt.

## Entfernte Bestandteile und ihre frühere Aufgabe

Die folgenden Erweiterungen waren im Entwicklungszweig vorhanden, wurden in
`main` aber bei der Eingrenzung auf den ReID-Versuch entfernt. Sie werden mit
dieser Integration nicht wieder in den aktiven Pfad aufgenommen.

| Bestandteil | Frühere Aufgabe | Status/Grund |
|---|---|---|
| Qdrant Vector Store | Profile und Embeddings alternativ zu SQLite lokal oder über einen Qdrant-Dienst speichern und durchsuchen | Nicht benötigt für den kontrollierten lokalen ReID-Versuch; würde ein zweites Persistenz-/Betriebsmodell einführen |
| Store Factory | Je nach Konfiguration SQLite oder Qdrant erzeugen | Mit Qdrant aus dem aktiven Umfang entfernt |
| Football Orchestrator | Spieler-ReID, Ball-, Team-, Spielfeld- und Statistikmodule koordinieren | Anwendungsdomäne außerhalb des abgegrenzten Personen-ReID-Versuchs |
| Ball Detector | Ballposition pro Frame erkennen und dem Analysefluss bereitstellen | Nur für Fußballanalyse relevant |
| Team Classifier | Spieler anhand von Trikot-/Farbinformationen Teams zuordnen | Kein Identitätsmerkmal der allgemeinen ReID; domänenspezifisch |
| Pitch Mapper | Bildkoordinaten per Kalibrierung auf Spielfeldkoordinaten abbilden | Benötigt Kamerakalibrierung/Homographie und Fußballfeldannahmen |
| Stats Aggregator | Laufstrecke, Geschwindigkeit, Team-/Ballereignisse und weitere Statistiken sammeln | Baut auf Pitch Mapping und Fußballereignissen auf |
| Football Mode | Gemeinsame Flags und Preset für diese Module bereitstellen | Entfernt, da die Module im Hauptpfad nicht vollständig verdrahtet waren |
| Max-Accuracy Mode | Größere Modelle, höhere Auflösung und aggressive Genauigkeitseinstellungen bündeln | Kein sauber isolierter Methodenvergleich zu B0/A1/A2/A3 |
| Allgemeine Motion-Diagnostik | Richtung, Pixelgeschwindigkeit, Sprünge und Plausibilität pro Tracker-ID protokollieren | Die umfangreiche Diagnose bleibt entfernt; nur der eng begrenzte Kontinuitätsbonus wurde übernommen |
| Model Manager und Windows-Setuphelfer | Modelle installieren und Start-/Umgebungsprüfungen vereinfachen | Automatische Installation bleibt entfernt; eine kleine, rein lokale Modell-Dateierkennung für die UI wurde wieder aufgenommen |

## Methodische Grenzen

- Detailmerkmale sind heuristisch und bei kleinen, unscharfen oder verdeckten
  Crops unzuverlässig.
- Veränderliche Merkmale wie Kappe, Uhr, Rucksack oder Aufdruck werden bei
  Widerspruch stark abgewichtet.
- Weak-Zuordnungen dürfen das gespeicherte Profil nicht aktualisieren.
- Der Motion-Bonus verwendet nur Bildraum-Nähe und kurze Zeitabstände. Er darf
  keinen schlechten visuellen Match über die Strong-Schwelle zwingen, wenn die
  konfigurierten Grenzen dies nicht zulassen.
- Einzelpersonen-Kennzahlen sind nicht auf Mehrpersonen-Videos übertragbar.

## Kommandozeile

```powershell
python scripts/evaluate_single_person_videos.py init-manifest --video-root Test-daten
python scripts/evaluate_single_person_videos.py run `
  --manifest data/evaluation_manifest.csv `
  --test-mode fixed_db `
  --preset details_tracking `
  --max-frames 0
```

`--preset default` führt denselben Einzelpersonentest mit der Main-Pipeline aus;
auch die übrigen eingebauten und selbst gespeicherten Presets sind zulässig.

Für die gemeinsamen Mehrpersonen-Szenarien G1-G4 und den automatischen Lauf
über B0/A1/A2/A3/D1-D6 siehe [G1_G4_EVALUATION.md](G1_G4_EVALUATION.md).
