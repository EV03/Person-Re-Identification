# Änderungsbericht – Detail Recognition v2

## Ziel

Die Detail-Erkennung wurde von einer kleinen Beispiel-Liste auf eine nachvollziehbare Detail-Schicht erweitert. Sie ersetzt die bestehende ReID über OSNet nicht, sondern liefert zusätzliche weiche Matching-Signale.

## Neue Idee

Bisher wurden nur wenige binäre Attribute betrachtet:

- Brille / keine Brille
- Kappe / keine Kappe
- Uhr / keine Uhr

Jetzt werden Details über eine Registry verwaltet. Dadurch ist nachvollziehbar:

- welches Detail genutzt wird,
- aus welcher Körperregion es stammt,
- ob es binär oder ein Vektor ist,
- wie stark es gewichtet wird,
- ob es volatil/veränderbar ist,
- wie stark es den Match beeinflusst.

## Neue Detail-Gruppen

### Kopf

- Brille
- Kappe/Mütze
- Kapuze
- längere Haare
- Gesichtsbehaarung
- Kopf-/Haarregion als Farb-Histogramm

### Oberkörper

- Oberkörperfarbe
- T-Shirt/Pullover/Jacke über Farbhistogramm
- T-Shirt-Aufdruck/Muster über Textur-/Kantenprofil
- Oberkörper-Region über kompaktes Strukturprofil

### Unterkörper

- Unterkörperfarbe, z. B. Hose/Rock über Farbhistogramm

### Accessoires

- Uhr
- Rucksack/Tasche

## Technische Umsetzung

Neue zentrale Registry in:

```text
app/utils/detail_utils.py
```

Beispielhafte Registry-Struktur:

```python
DetailSpec(
    name="upper_color_hist",
    label="Oberkörperfarbe",
    group="torso",
    kind="vector",
    weight=0.16,
    volatile=False,
    length=18,
    description="HSV color histogram of upper body / shirt / pullover / jacket",
)
```

## Matching-Logik

Die OSNet-Ähnlichkeit bleibt die Hauptentscheidung.

Details werden nur als Re-Ranking-Signal genutzt:

```text
final_score = visual_score + (detail_score - 0.5) * detail_weight
```

Bedeutung:

- `visual_score` ist die normale OSNet-Ähnlichkeit.
- `detail_score = 0.5` ist neutral.
- ähnliche Details geben einen kleinen Bonus.
- widersprüchliche Details geben einen kleinen Abzug.
- veränderbare Details zerstören den Match nicht automatisch.

## Nachvollziehbarkeit

In der Event-Tabelle werden zusätzliche Werte gespeichert:

- `match_visual_score`
- `match_detail_score`
- `match_detail_weight`
- `match_reason`
- `details_label`
- `details_binary`
- `detail_reliability`

Zusätzlich gibt es im Streamlit-UI unter **Detail Analysis** einen Expander:

```text
Welche Details werden genutzt?
```

Dort werden alle Registry-Einträge mit Gruppe, Typ, Gewicht, Volatilität und Beschreibung angezeigt.

## Grenzen der aktuellen Umsetzung

Die neue Version nutzt weiterhin leichte OpenCV-Heuristiken. Das ist bewusst so, damit die App lokal, klein und ohne zusätzliche Modellabhängigkeiten bleibt.

Nicht zuverlässig genug für produktive Detail-Erkennung sind aktuell:

- echte Logo-/Schrifterkennung,
- genaue Bart-/Haar-Klassifikation,
- sichere Uhr-Erkennung,
- sichere Unterscheidung von Hoodie/Pullover/Jacke,
- robuste Detailerkennung bei sehr kleinen Person-Crops.

Für höhere Genauigkeit sollte später ein trainierter Attribut-Classifier oder eine Pose-/Keypoint-basierte Aufteilung ergänzt werden.

## Betroffene Dateien

```text
app/utils/detail_utils.py
app/storage/models.py
app/storage/vector_store.py
app/storage/qdrant_vector_store.py
app/pipeline/orchestrator.py
app/ui/streamlit_app.py
docs/detail_recognition_v2_changelog.md
```
