# ReID-/Tracking-Verbesserungen – aktueller Stand und spätere Optionen

## Ziel

Das aktuelle Ziel ist nicht zuerst ein größeres YOLO-Modell oder ein anderer Tracker, sondern eine stabilere Speicherung und Aktualisierung der ReID-Embeddings.

Die Pipeline soll weniger schlechte Bilder/Crops in die Datenbank aufnehmen. Dadurch sollen weniger falsche Personen entstehen und bestehende Personen-Embeddings nicht durch unscharfe, zu dunkle, abgeschnittene oder zu kleine Crops verschlechtert werden.

## Aktuelle Entscheidung

Folgende Verbesserungen wurden aktuell priorisiert:

1. **ReID-Frequenz reduziert auf alle 5 Frames**
   - `reid_every_n_frames = 5`
   - Dadurch werden häufiger gute Kandidaten gefunden als vorher bei 10 Frames.

2. **Embeddings werden nicht mehr sofort beim ersten Track gespeichert**
   - Neue Tracks werden erst dann gematcht oder als neue Person gespeichert, wenn eine Mindestanzahl guter Crops gesammelt wurde.
   - Standardwert: `min_good_frames_before_reid = 3`

3. **Quality Gate für ReID-Crops**
   - Schlechte Crops werden nicht encodiert und nicht gespeichert.
   - Bewertet werden unter anderem:
     - Schärfe / Blur
     - Helligkeit
     - Crop-Größe
     - Seitenverhältnis
     - Randkontakt / abgeschnittene Person
     - Detection Confidence

4. **Personen-Embeddings werden nur mit guten Crops aktualisiert**
   - Ein bestehendes `mean_embedding` wird nur noch aktualisiert, wenn der Crop mindestens die Update-Qualität erfüllt.
   - Standardwerte:
     - `min_embedding_quality = 0.55`
     - `min_update_quality = 0.65`

5. **Embedding-Updates werden qualitätsgewichtet**
   - Gute Crops gehen stärker in das `mean_embedding` ein.
   - Schwächere, aber noch akzeptierte Crops haben weniger Einfluss.

6. **Bestes Snapshot-Bild wird anhand der Qualität aktualisiert**
   - Pro Person wird nicht nur irgendein erstes Bild behalten.
   - Wenn später ein besserer Crop entsteht, wird dieser als `best_snapshot_path` gespeichert.

## Warum diese Änderung wichtig ist

Vorher konnte die Pipeline beim ersten Auftauchen einer Track-ID sofort ein Embedding erzeugen und speichern. Das ist problematisch, weil der erste Crop oft nicht der beste ist:

- Person ist gerade nur halb im Bild.
- Person ist unscharf.
- Person ist zu klein.
- Person ist durch Bewegung verwischt.
- Person wird am Bildrand abgeschnitten.
- Box enthält zu viel Hintergrund.

Dadurch kann die Datenbank verschlechtert werden. Besonders kritisch ist das Aktualisieren bestehender Personen-Embeddings: Wenn schlechte Crops in den Mittelwert eingehen, wird das gespeicherte Personenprofil ungenauer.

Die neue Logik ist deshalb:

```text
Track erkannt
→ Crop erstellen
→ Qualität prüfen
→ gute Crops sammeln
→ erst nach mehreren guten Frames matchen/speichern
→ bestehende Person nur mit guten Crops aktualisieren
```

## Zurückgestellte, aber dokumentierte spätere Optionen

Folgende Möglichkeiten bleiben sinnvoll, werden aber aktuell bewusst nach hinten gestellt:

### 1. Stärkeres YOLO-Modell

Mögliche Upgrades:

```text
yolov8n.pt → yolov8s.pt → yolov8m.pt
```

Vorteil:

- stabilere Personenerkennung
- bessere Bounding Boxes
- weniger schlechte Crops

Nachteil:

- höhere Laufzeit
- mehr GPU-/CPU-Last

### 2. Tracking-Algorithmus wechseln oder anpassen

Mögliche Optionen:

```text
bytetrack.yaml → botsort.yaml
```

Zusätzlich später möglich:

- eigenes Tracker-YAML
- Track Buffer anpassen
- Match Threshold im Tracker anpassen
- Kamera-Bewegungsausgleich / GMC testen
- BoT-SORT mit Appearance-/ReID-Unterstützung prüfen

### 3. Höhere Bildgröße / bessere Aufnahmequalität

Mögliche Anpassungen:

```text
image_size = 960
image_size = 1280
```

Zusätzlich:

- bessere Beleuchtung
- weniger Bewegungsunschärfe
- stabilere Kamera
- weniger Perspektivwechsel
- höhere Videoauflösung

### 4. Bewegungsrichtung und Plausibilitätsprüfung

Später kann pro Track zusätzlich gespeichert werden:

```text
center_x
center_y
velocity_x
velocity_y
last_bbox
last_seen_frame
```

Damit kann geprüft werden, ob eine neue Box realistisch zur bisherigen Bewegungsrichtung passt.

### 5. Segmentierung und Körperteile

Spätere Erweiterungen:

- Personensegmentierung statt rechteckiger Bounding Box
- Oberkörper-/Unterkörper-Embeddings
- Full-Body + Torso-Embedding
- Pose-/Keypoint-basierte Körperanalyse
- Gesichtserkennung nur optional und mit besonderem Datenschutzfokus
- Bewegungsmuster als ergänzendes Signal

Diese Punkte sind sinnvoll, aber nicht der nächste MVP-Schritt. Zuerst wird die vorhandene Pipeline robuster gemacht.

## Neue Standardparameter

```text
reid_every_n_frames = 5
min_good_frames_before_reid = 3
min_embedding_quality = 0.55
min_update_quality = 0.65
```

## Betroffene Dateien

Vollständig zu ersetzen:

```text
app/config.py
app/modes/base_mode.py
app/modes/default_mode.py
app/modes/football_mode.py
app/pipeline/orchestrator.py
app/storage/vector_store.py
app/ui/streamlit_app.py
app/utils/image_utils.py
```

Neu:

```text
docs/reid_tracking_verbesserungen.md
```

Optional zu ersetzen/aktualisieren:

```text
Regeln und Zustand.md
```
