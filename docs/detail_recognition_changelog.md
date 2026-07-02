# Änderungsbericht – Detail-Erkennung / Attribute Layer

## Ziel

Die bestehende Person-ReID-Pipeline wurde um eine zusätzliche Detail-Erkennung erweitert. Neben dem vollständigen Personen-Embedding werden nun weiche Detail-Signale aus dem Person-Crop extrahiert, z. B.:

- Brille / keine Brille,
- Kappe / keine Kappe,
- Uhr / keine Uhr.

Dadurch soll ein einzelnes verändertes Detail, z. B. eine abgesetzte Kopfbedeckung, die Wiedererkennung nicht allein dominieren. Details werden nur als kleine zusätzliche Matching-Schicht verwendet und ersetzen nicht das OSNet-ReID-Embedding.

## Technisches Konzept

Neue Pipeline-Logik:

```text
YOLO Person Detection
→ ByteTrack / BoT-SORT Tracking
→ Person Crop
→ Crop Quality Gate
→ OSNet Embedding
→ DetailFeatureExtractor
→ Visual Match Score
→ Detail Similarity Score
→ Soft Combined Score
→ SQLite/Qdrant speichern
→ Live Label + Event-Tabelle
```

## Matching-Logik

Der OSNet-Score bleibt der Hauptscore. Die Detail-Erkennung verändert den Score nur leicht:

```text
final_score = visual_score + (detail_score - 0.5) * detail_weight
```

Bedeutung:

- `detail_score = 0.5` ist neutral.
- Passende Details erhöhen den Score leicht.
- Widersprüchliche Details senken den Score leicht.
- Ein einzelnes Detail wie Kappe ja/nein kann den Gesamtmatch nicht allein zerstören.

## Neue UI-Optionen

Im Streamlit-Sidebar-Bereich wurde ergänzt:

- `Enable detail recognition layer`
- `Detail matching weight`
- `Min detail confidence`
- `Draw detail labels`

## Neue Speicherung

Die SQLite-Tabelle `persons` wurde erweitert um:

- `detail_vector`
- `detail_state_json`
- `detail_weight_sum`

Die Event-Tabelle zeigt zusätzlich:

- `details_label`
- `detail_reliability`

Qdrant speichert Detaildaten zusätzlich im Point-Payload und nutzt sie beim Re-Ranking.

## Betroffene Dateien

```text
app/config.py
app/modes/base_mode.py
app/modes/default_mode.py
app/modes/max_accuracy_mode.py
app/pipeline/orchestrator.py
app/storage/qdrant_vector_store.py
app/storage/vector_store.py
app/ui/streamlit_app.py
app/utils/detail_utils.py
app/utils/image_utils.py
docs/detail_recognition_changelog.md
```

## Wichtige Einschränkung

Die aktuelle Detail-Erkennung ist eine lokale, leichte MVP-Implementierung mit OpenCV-Heuristiken. Sie ist bewusst ohne neue große Modellabhängigkeiten umgesetzt. Für verlässliche Detail-Erkennung in produktiven Tests sollte dieser Layer später durch trainierte Attribute-Classifier, Pose-/Keypoint-Analyse oder segmentierte Körperbereiche ersetzt werden.

## Neuer Stand

Die Pipeline kann jetzt neben Gesamtpersonen-Embeddings auch Detailmerkmale erfassen, speichern, im UI anzeigen und als weiches Re-Ranking-Signal für Person-ReID verwenden.
