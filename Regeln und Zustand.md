# Regeln und Zustand – Person-Re-Identification-Projekt

## 1. Projektziel

Ziel des Projekts ist der Aufbau eines möglichst kleinen, lokal lauffähigen **Person-Re-Identification-MVPs**.

Das System soll:

- eine Kamera oder ein Video als Eingabe verwenden,
- Personen im Bild erkennen,
- erkannte Personen live tracken,
- aus jeder Person visuelle Merkmale extrahieren,
- diese Merkmale als Embeddings speichern,
- bekannte Personen bei erneuter Sichtung wiedererkennen,
- Ergebnisse live visualisieren,
- lokal und möglichst kostenlos laufen.

Die ursprüngliche Idee eines „LLM zur Personenerkennung“ wurde in eine passendere **Computer-Vision-Pipeline** überführt.

---

## 2. Geplante Architektur

```mermaid
flowchart LR
    A[Kamera / Video] --> B[YOLO Person Detection]
    B --> C[ByteTrack / BoT-SORT Tracking]
    C --> D[Person Crop]
    D --> E[ReID Modell / OSNet]
    E --> F[Embedding Vector]
    F --> G[Vector Store / SQLite]
    G --> H[Known Person oder New Person]
    H --> I[Live Preview + Datenbank]
```

Die Kernbestandteile:

| Bereich | Geplante Komponente |
|---|---|
| UI | Streamlit |
| Kamera / Video | OpenCV |
| Personenerkennung | YOLOv8 |
| Tracking | ByteTrack / BoT-SORT |
| Re-Identification | Torchreid OSNet |
| Demo-Embedding | ColorHistogram |
| Speicherung | SQLite |
| spätere Vector DB | Qdrant optional |
| Deployment | lokale Python-Umgebung |

---

## 3. Was umgesetzt wurde

### 3.1 Lokales MVP erstellt

Es wurde ein ZIP-Projekt mit notwendiger Grundstruktur erstellt:

```text
person-reid-mvp/
├── app/
│   ├── config.py
│   ├── ui/
│   │   └── streamlit_app.py
│   ├── pipeline/
│   │   ├── detector_tracker.py
│   │   ├── orchestrator.py
│   │   └── reid_encoder.py
│   ├── storage/
│   │   ├── metadata_store.py
│   │   └── vector_store.py
│   └── utils/
├── scripts/
├── data/
├── requirements.txt
├── requirements-optional-reid.txt
├── docker-compose.yml
└── README.md
```

---

### 3.2 Streamlit-Eingabepanel umgesetzt

Das UI ermöglicht inzwischen:

- Video-Upload,
- lokale Webcam-Nutzung,
- Auswahl der Kameraquelle,
- Testbild der Kamera,
- Start der Re-Identification,
- Live-Anzeige der erkannten Personen,
- Anzeige von Match-Daten,
- Ausgabe von Datenbankinformationen.

---

### 3.3 Kameraauswahl verbessert

Anfangs wurde nur OpenCV-Kameraindex `0` verwendet.

Danach wurde ergänzt:

- Kamera-Scan,
- manuelle Kameraindex-Auswahl,
- Auswahl verschiedener Kamera-Backends:
  - `dshow`,
  - `msmf`,
  - `auto`,
- Kamera-Testbild vor Start der Erkennung.

Damit kann die laufende Webcam besser ausgewählt und getestet werden.

---

### 3.4 Live-Preview ergänzt

Ursprünglich wurden Personen zwar erkannt und gespeichert, aber nicht live visualisiert.

Danach wurde ergänzt:

- Live-Bild im Streamlit-Panel,
- Bounding Boxes auf dem echten Kamerabild,
- Anzeige von:
  - Track-ID,
  - Person-ID,
  - Match-Score,
  - Detection-Confidence.

Damit ist jetzt sichtbar, welche Person erkannt und wie sie getrackt wird.

---

### 3.5 Detection und Tracking funktionieren

Aktuell läuft:

```text
YOLOv8n + ByteTrack
```

Bedeutung:

- YOLOv8n erkennt Personen im Kamerabild.
- ByteTrack hält temporäre Track-IDs während der laufenden Aufnahme.
- Die erkannte Person wird als Crop weiterverarbeitet.

YOLO und ByteTrack mussten nicht separat installiert werden, weil sie über `ultralytics` bereitgestellt werden.

---

### 3.6 Re-Identification funktioniert

Zuerst lief ein einfacher Demo-Encoder:

```text
ColorHistogramEncoder
```

Dieser erzeugt 32-dimensionale Embeddings auf Basis von Farbverteilungen.

Danach wurde Torchreid angebunden:

```text
Torchreid OSNet
```

Aktuell wurde OSNet erfolgreich geladen:

```text
Successfully loaded imagenet pretrained weights
Model: osnet_x1_0
params: 2,193,616
flops: 978,878,352
```

Das bedeutet: Das echte ReID-Modell läuft.

---

## 4. Fehlerbehebung

### 4.1 Torchreid nicht richtig importierbar

Fehler:

```text
ModuleNotFoundError: No module named 'gdown'
```

Lösung:

```powershell
python -m pip install gdown yacs h5py Cython scipy tensorboard scikit-learn imageio tqdm
```

---

### 4.2 Falsche Torchreid-Paketstruktur

Das zuerst installierte Paket hatte nicht die erwartete Struktur:

```text
No module named 'torchreid.utils'
```

Ursache:

```text
torchreid-pip war nicht identisch mit der erwarteten Original-Torchreid-Struktur
```

Es wurde geprüft, welche Imports funktionieren und wie das Paket korrekt eingebunden werden muss.

---

### 4.3 Embedding-Dimensionen passten nicht zusammen

Fehler:

```text
shapes (512,) and (32,) not aligned: 512 (dim 0) != 32 (dim 0)
```

Ursache:

- Alte Datenbank enthielt 32-dimensionale `colorhist`-Embeddings.
- Neues OSNet erzeugt 512-dimensionale Embeddings.

Lösung:

- Datenbank zurücksetzen.
- Danach nur noch neue 512-dimensionale OSNet-Embeddings speichern.

Nach dem Reset funktionierte die Re-Identification im ersten Versuch.

---

## 5. Aktueller Stand

Das Projekt ist aktuell ein funktionierender lokaler MVP.

| Bereich | Status |
|---|---|
| Streamlit UI | funktioniert |
| Webcam-Test | funktioniert |
| Kameraauswahl | funktioniert |
| Live-Kamera-Preview | funktioniert |
| Person Detection | funktioniert |
| Tracking | funktioniert |
| Re-Identification | funktioniert |
| Torchreid / OSNet | läuft |
| Embedding-Speicherung | funktioniert |
| SQLite-Datenbank | funktioniert |
| Live-Visualisierung | funktioniert |
| Fehler durch alte Embeddings | behoben |

Aktuelle Pipeline:

```text
Live-Kamera
→ YOLOv8n erkennt Person
→ ByteTrack trackt Person
→ Person-Crop wird erstellt
→ Torchreid OSNet erzeugt 512D-Embedding
→ SQLite Vector Store sucht ähnliche Person
→ bekannte Person-ID oder neue Person-ID
→ Live-Bild visualisiert Erkennung
```

---

## 6. Aktuelle Modellentscheidung

Aktuell wird dieses Visual-Computing-Setup verwendet:

| Aufgabe | Modell / Tool |
|---|---|
| Personenerkennung | YOLOv8n |
| Tracking | ByteTrack |
| ReID-Modell | Torchreid OSNet `osnet_x1_0` |
| Embedding-Größe | 512 |
| Datenbank | SQLite |
| UI | Streamlit |

SAM/SAM3 wurde diskutiert, aber nicht als ReID-Modell verwendet.

Begründung:

- SAM ist primär für Segmentierung und Objektmasken geeignet.
- Re-Identification braucht Embeddings.
- OSNet ist dafür passender.

SAM könnte später optional als Masken-/Segmentierungsmodul vor dem ReID-Encoding ergänzt werden.

---

## 7. Bisherige Arbeitsregeln

### Regel 1: Kein LLM für visuelle Personenerkennung

Personenerkennung und Wiedererkennung werden nicht über ein LLM gelöst, sondern über Computer Vision.

```text
Detection → Tracking → ReID Embedding → Vector Search
```

---

### Regel 2: Lokal, kostenlos und klein starten

Das Setup soll möglichst klein bleiben:

- lokale Ausführung,
- keine Cloud-Pflicht,
- keine kostenpflichtigen APIs,
- schneller MVP-Aufbau.

---

### Regel 3: MVP zuerst, Optimierung später

Zuerst wird eine funktionierende Pipeline gebaut.

Danach werden einzelne Komponenten verbessert:

```text
erst lauffähig
dann stabiler
dann schneller
dann genauer
```

---

### Regel 4: ReID ist nicht gleich Tracking

Tracking und Re-Identification werden getrennt betrachtet:

| Begriff | Bedeutung |
|---|---|
| Tracking | gleiche Person innerhalb eines laufenden Videos verfolgen |
| ReID | Person über Zeit, Neustarts oder mehrere Videos wiedererkennen |

---

### Regel 5: Keine echten Namen oder personenbezogenen Profile speichern

Das System arbeitet mit synthetischen IDs:

```text
person_000001
person_000002
```

Keine Klarnamen, keine unnötigen personenbezogenen Informationen.

---

### Regel 6: Embedding-Dimensionen dürfen nicht gemischt werden

Wenn der Encoder gewechselt wird, muss die Datenbank zurückgesetzt oder migriert werden.

| Wechsel | Reset nötig |
|---|---|
| ColorHistogram 32D → OSNet 512D | ja |
| OSNet 512D → OSNet 512D | nein |

---

### Regel 7: Starten immer über aktive `.venv`

Die App soll über die aktive virtuelle Umgebung gestartet werden:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/ui/streamlit_app.py
```

Nicht bevorzugt:

```powershell
streamlit run app/ui/streamlit_app.py
```

Grund: Dadurch kann versehentlich ein globales Python/Streamlit verwendet werden.

---

### Regel 8: Fehler erst isoliert prüfen

Bei Import- oder Laufzeitproblemen wird zuerst einzeln geprüft:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torchreid; print(torchreid.__file__)"
```

Danach wird erst der App-Code angepasst.

---

### Regel 9: Änderungen werden als Änderungsbericht mitgeführt

Ab jetzt soll bei weiteren Änderungen aktiv dokumentiert werden:

```text
Was wurde geändert?
Warum wurde es geändert?
Welche Dateien sind betroffen?
Welcher Fehler wurde gelöst?
Wie ist der neue Stand?
```

---

### Regel 10: Bericht per Token abrufbar

Wenn im Chat geschrieben wird:

```text
!bericht
```

soll der aktuelle Projektbericht zurückgegeben werden.

Der Bericht soll enthalten:

- geplantes Ziel,
- bisherige Umsetzung,
- aktueller Stand,
- technische Architektur,
- gelöste Probleme,
- offene Punkte,
- aktuelle Regeln,
- letzte Änderungen.

---

## 8. Offene Punkte / nächste sinnvolle Schritte

### 8.1 Sicherer Debug-Status im UI

Im UI sollte sichtbar sein:

```text
Detection Model: yolov8n.pt
Tracker: bytetrack.yaml
ReID Encoder: torchreid / OSNet
Embedding Dimension: 512
Device: cpu/cuda
Database: active
```

---

### 8.2 Datenbank-Reset im UI

Aktuell wurde der Reset über Script oder manuell gemacht.

Sinnvoll wäre ein Button:

```text
Reset ReID Database
```

mit Warnhinweis.

---

### 8.3 Schutz gegen falsche Embedding-Dimensionen

Der Vector Store sollte alte inkompatible Embeddings automatisch ignorieren, statt abzustürzen.

Beispiel:

```text
stored=(32,), query=(512,) → skip
```

---

### 8.4 GPU-Nutzung sauber anzeigen

Das UI sollte anzeigen:

```text
Torch CUDA available: True/False
Current device: cuda/cpu
GPU name: ...
```

---

### 8.5 Modelloptionen erweitern

Mögliche spätere Varianten:

| Bereich | Option |
|---|---|
| Detection | YOLOv8s statt YOLOv8n |
| Tracking | BoT-SORT testen |
| ReID | anderes OSNet-Modell |
| Vector DB | Qdrant statt SQLite |
| Segmentierung | SAM/SAM2/SAM3 optional |

---

### 8.6 Bessere Personenverwaltung

Sinnvoll wäre später:

- Person-ID-Liste,
- gespeicherte Snapshots anzeigen,
- falsche Matches löschen,
- Personen zusammenführen,
- Personen manuell zurücksetzen,
- ReID-Historie pro Person anzeigen.

---

## 9. Zusammenfassung

Das Projekt hat inzwischen einen funktionierenden lokalen MVP erreicht.

Gestartet wurde mit der Idee eines lokalen KI-/LLM-Systems zur Personenerkennung. Daraus wurde eine technisch passende Computer-Vision-Architektur entwickelt.

Aktuell läuft:

```text
YOLOv8n + ByteTrack + Torchreid OSNet + SQLite + Streamlit Live UI
```

Die Kameraerkennung, Personenerkennung, Live-Visualisierung und Re-Identification funktionieren. Der wichtigste technische Durchbruch war die erfolgreiche Umstellung von einfachen 32D-Farb-Embeddings auf echte 512D-OSNet-ReID-Embeddings.

---

## 10. Änderungsbericht

### Aktueller letzter Stand

- Re-Identification funktioniert nach Datenbank-Reset mit OSNet.
- OSNet erzeugt 512-dimensionale Embeddings.
- Alte 32D-ColorHistogram-Daten waren inkompatibel und wurden entfernt.
- Die Live-Erkennung und Visualisierung funktionieren.
- Der Bericht wurde als Markdown-Datei `Regeln und Zustand.md` erstellt.

### Neue Arbeitsregel

Ab jetzt wird bei weiteren Änderungen aktiv ein Änderungsbericht mitgeführt. Wenn das Token `!bericht` geschrieben wird, wird der aktuelle Projektbericht in den Chat zurückgegeben.

---

## 11. Neue Änderung – Mode-System und Football-Analysemodus

### 11.1 Ziel der Änderung

Der bisherige lauffähige MVP wird als eigener Modus mit dem Namen `default` geführt.
Zusätzlich wurde ein weiterer auswählbarer Modus für die langfristige Fußballanalyse vorbereitet:

```text
football_team_analysis
```

Langfristiges Ziel dieses Modus ist die Verarbeitung von Videoeingaben, um Spieler einer Fußballmannschaft zu erkennen, zu tracken und daraus Statistiken abzuleiten.

---

### 11.2 Umgesetzte Änderungen

Es wurde ein neues Mode-System eingeführt:

```text
app/modes/
├── base_mode.py
├── default_mode.py
├── football_mode.py
└── mode_registry.py
```

Der aktuelle Projektstand ist jetzt als `default` registriert.

Der neue Fußballmodus ist als `football_team_analysis` registriert und über das Streamlit-UI auswählbar.

Zusätzlich können im UI eigene Custom Modes erstellt und gespeichert werden. Diese werden lokal gespeichert unter:

```text
data/modes/custom_modes.json
```

---

### 11.3 UI-Änderungen

In `app/ui/streamlit_app.py` wurde ergänzt:

- Modusauswahl in der Sidebar,
- Anzeige der Modusbeschreibung,
- Formular zum Erstellen eigener Custom Modes,
- modeabhängige Default-Werte für YOLO, Tracker, Encoder, Threshold und Max Frames,
- Anzeige von Analyse-Läufen,
- Ausgabe des aktuell verwendeten Modus im Run Summary.

---

### 11.4 Pipeline-Änderungen

In `app/pipeline/orchestrator.py` wurde ergänzt:

- `mode_id`, `mode_name` und `pipeline_type` werden verarbeitet,
- Output-Videos enthalten den Modus im Dateinamen,
- Analyse-Läufe werden in SQLite dokumentiert,
- Events speichern die Mode-Information im Payload,
- Football Mode gibt aktuell einen Warnhinweis aus, dass die Fußballmodule vorbereitet, aber noch nicht vollständig verdrahtet sind.

---

### 11.5 Datenbank-Erweiterung

In `app/storage/vector_store.py` wurden neue Tabellen vorbereitet:

```text
analysis_runs
teams
players
player_frame_events
ball_frame_events
player_stats
```

Damit ist die Grundlage für spätere Fußballstatistiken vorhanden.

Außerdem wurde der Schutz gegen inkompatible Embedding-Dimensionen ergänzt:

```text
stored=(32,), query=(512,) → skip
```

Dadurch stürzt die Suche nicht mehr ab, wenn alte ColorHistogram-Daten und neue OSNet-Daten gemischt vorliegen.

---

### 11.6 Football-Module vorbereitet

Es wurden vorbereitende Module ergänzt:

```text
app/pipeline/football/
├── ball_detector.py
├── football_orchestrator.py
├── pitch_mapper.py
├── stats_aggregator.py
└── team_classifier.py
```

Aktueller Stand:

| Modul | Status |
|---|---|
| Ball Detector | Platzhalter |
| Team Classifier | einfache Farblogik vorbereitet |
| Pitch Mapper | Homography-Grundlage vorbereitet |
| Stats Aggregator | Distanz, Geschwindigkeit, Sprint-Logik vorbereitet |
| Football Orchestrator | Erweiterungspunkt vorbereitet |

---

### 11.7 Neue Arbeitsregel

Bei neuen fachlichen Analysemodi wird zuerst ein eigener Modus mit stabiler Default-Pipeline erstellt. Erst danach werden neue Fachmodule schrittweise verdrahtet.

Begründung:

```text
bestehender MVP bleibt stabil
neuer Modus kann iterativ erweitert werden
Fehler bleiben besser isolierbar
```

---

### 11.8 Neuer aktueller Stand

Aktuell existieren zwei Built-in-Modes:

| Mode ID | Name | Status |
|---|---|---|
| `default` | Default ReID MVP | produktiver aktueller MVP-Stand |
| `football_team_analysis` | Football Team Analysis | auswählbare Hülle mit vorbereiteten Football-Modulen |

Der Football Mode nutzt in V1 noch die bestehende ReID-Pipeline. Die nächsten sinnvollen Schritte sind:

1. Team-Farbklassifikation in die Crop-Verarbeitung einbauen.
2. Ball-Erkennung mit eigenem Modell oder Roboflow-Dataset ergänzen.
3. Pitch-Mapping per Homography aktivieren.
4. Spielerpositionen in Meterkoordinaten speichern.
5. Statistiken aus den gespeicherten Positionsdaten berechnen.
