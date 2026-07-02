# ReID Matching, Kalibrierung und kontrolliertes Profilwachstum

## Ziel

Diese Änderung verschiebt die Pipeline weg von der einfachen Regel:

```text
Score >= Threshold -> Match
Score < Threshold  -> neue Person
```

hin zu einer dreistufigen ReID-Entscheidung:

```python
if score >= strong_match_threshold:
    assign_to_existing_person()
elif score >= weak_match_threshold:
    assign_as_candidate_or_pending()
else:
    wait_for_more_frames_before_new_person()
```

Dadurch soll verhindert werden, dass einzelne schlechte Crops oder kurze Perspektivwechsel sofort neue endgültige Personenprofile erzeugen.

## Neue Standardparameter

### Allgemeiner Test-/Videobetrieb

```python
match_threshold = 0.74
strong_match_threshold = 0.82
weak_match_threshold = 0.68
new_person_max_score = 0.58
new_person_min_evidence_events = 6
new_person_evidence_window_frames = 30
new_person_low_match_ratio = 0.80

detection_confidence = 0.40
image_size = 960
reid_every_n_frames = 5
min_good_frames_before_reid = 3
min_embedding_quality = 0.60
min_update_quality = 0.75
min_crop_height = 120
min_crop_width = 45
crop_padding = 0.08

detail_weight = 0.05
detail_min_confidence = 0.70
```

### Kalibrierungsmodus

Kalibrierung soll ein hochwertiges Startprofil erzeugen, aber nicht als einzige Wahrheit gelten.

```python
calibration_detection_confidence = 0.45
calibration_image_size = 1280
calibration_reid_every_n_frames = 3
calibration_min_good_frames_before_reid = 5
calibration_min_embedding_quality = 0.70
calibration_min_update_quality = 0.80
calibration_min_crop_height = 160
calibration_min_crop_width = 60
calibration_crop_padding = 0.10
calibration_enable_detail_analysis = False
```

## Neue Matching-Logik

### Strong Match

Ein Match ab `strong_match_threshold` gilt als sicher.

Folgen:

- vorhandene Person-ID wird übernommen,
- Profil darf wachsen,
- Embedding wird als Sample gespeichert,
- Ereignis wird als `matched_person` geloggt.

### Weak / Pending Match

Ein Match zwischen `weak_match_threshold` und `strong_match_threshold` wird als plausibel, aber unsicher behandelt.

Folgen:

- es wird keine neue Person erstellt,
- die plausible bestehende Person-ID wird temporär übernommen,
- das Profil wird zunächst nicht aktualisiert,
- der Track wird erneut geprüft,
- Ereignis wird als `pending_weak_match` geloggt.

### Low Match / potenziell neue Person

Liegt der Score unter `weak_match_threshold`, entsteht nicht sofort eine neue Person.

Eine neue Person wird erst erzeugt, wenn innerhalb eines Frame-Fensters genügend Low-Match-Evidenz gesammelt wurde:

```python
new_person_min_evidence_events = 6
new_person_evidence_window_frames = 30
new_person_low_match_ratio = 0.80
new_person_max_score = 0.58
```

Dadurch muss ein Track über mehrere gute ReID-Events hinweg eher zu keiner bekannten Person passen, bevor ein neues Profil entsteht.

## Einzelne Embedding-Samples

Zusätzlich zum bisherigen `mean_embedding` werden neue Embeddings jetzt separat gespeichert:

```text
person_embedding_samples
```

Dadurch kann die Suche nicht nur gegen einen gemittelten Vektor laufen, sondern auch gegen einzelne hochwertige Referenzen einer Person. Das ist wichtig, weil dieselbe Person je nach Perspektive, Kleidung, Licht und Bewegung sehr unterschiedliche Embeddings erzeugen kann.

Die Suche bewertet deshalb:

```text
mean embedding
+ einzelne sample embeddings
```

und nimmt pro Person den besten Treffer.

## Top-K-Matching

Die SQLite-Suche liefert jetzt mehrere Kandidaten zurück. Events enthalten dadurch zusätzliche Informationen wie:

```text
top_matches
decision_zone
match_visual_score
match_detail_score
reference_type
reference_quality
```

Damit kann später nachvollzogen werden, ob die erwartete Person-ID komplett falsch war oder nur knapp gegen eine andere ID verloren hat.

## Details konservativer genutzt

Die Detail-Erkennung wird nicht entfernt, aber ihr Einfluss wurde reduziert:

- unsichere Details werden ignoriert,
- stabile Details können leicht positiv wirken,
- volatile Details wie Kappe, Kapuze, Uhr, Rucksack oder Aufdruck wirken nur schwach negativ,
- `detail_weight` wurde auf `0.05` reduziert,
- `detail_min_confidence` wurde auf `0.70` erhöht.

Damit sollen fehlerhafte Attribute wie `cap:yes` oder `hood:yes` das OSNet-Matching nicht mehr stark stören.

## Motion-/Positionsstützung

Die vorhandene Bewegungsanalyse wird zusätzlich als schwaches Plausibilitätssignal genutzt.

Wenn eine Person räumlich nah an einer vorherigen Position derselben Person erscheint, kann ein kleiner Bonus vergeben werden:

```python
motion_identity_bonus = 0.04
motion_identity_max_frame_gap = 15
motion_identity_max_distance_fraction = 0.15
```

Das ist kein Ersatz für ReID, sondern nur eine Stabilisierung gegen Track-Sprünge innerhalb eines Videos.

## Strikte Merge-Kandidaten-Werte

Für spätere Merge-Logik wurden strikte Parameter ergänzt:

```python
merge_candidate_threshold = 0.86
merge_candidate_min_events = 8
merge_candidate_same_track_required = True
```

Der aktuelle Patch führt noch keine automatische Personenverschmelzung aus. Die Werte sind vorbereitet, damit spätere Merge-Entscheidungen konservativ bleiben, besonders wenn mehrere Personen im Bild vorkommen.

## Streamlit-Frontend

Im Frontend gibt es neue Einstellmöglichkeiten:

- Parameter-Profil:
  - Standard/Video
  - Kurzvideo adaptiv
  - Langes Video stabil
  - Kalibrierung Qualität
- Three-zone ReID decision
- New-person evidence parameters
- Crop quality gate
- Qualitätswerte für Embedding und Update
- konservativere Detailwerte

## Evaluationsskript

Das Evaluationsskript akzeptiert jetzt zusätzliche Parameter, z. B.:

```powershell
python scripts/evaluate_single_person_videos.py run `
  --manifest "data\evaluation_manifest.csv" `
  --output-dir "data\evaluation_runs" `
  --test-mode adaptive_calibration `
  --match-threshold 0.74 `
  --strong-match-threshold 0.82 `
  --weak-match-threshold 0.68 `
  --new-person-max-score 0.58 `
  --image-size 960 `
  --calibration-image-size 1280 `
  --max-frames 0
```

Kalibrierungszeilen im Manifest nutzen automatisch die strengeren Kalibrierungsparameter.

## Erwarteter Effekt

Diese Änderung soll nicht sofort alle ReID-Probleme lösen, sondern die strukturell kritischsten Fehler reduzieren:

- zu frühe neue Personen,
- Identity Drift durch einzelne schlechte Crops,
- Übergewichtung fehlerhafter Details,
- fehlende Nachvollziehbarkeit der Matching-Entscheidung,
- Verlust von nützlichen Einzel-Embeddings durch reinen Mittelwert.
