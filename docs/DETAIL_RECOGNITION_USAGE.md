# Detail Recognition – Nutzung und Interpretation

## Was die Detail-Schicht macht

Die Detail-Schicht analysiert zusätzlich zum Gesamtpersonen-Embedding sichtbare Einzelmerkmale aus dem Person-Crop.

Sie dient nicht dazu, eine Person allein anhand eines Details zu erkennen. Sie soll Bias reduzieren, wenn ein Gesamtbild durch einzelne veränderbare Merkmale beeinflusst wird.

Beispiel:

```text
Person mit Kappe erkannt
Person nimmt Kappe ab
OSNet-Score sinkt leicht
Oberkörperfarbe, Unterkörperfarbe und Struktur bleiben ähnlich
Detail-Re-Ranking verhindert vorschnelle neue Person-ID
```

## Wie die Entscheidung gelesen wird

In der Event-Tabelle sind besonders wichtig:

| Spalte | Bedeutung |
|---|---|
| score | finaler Match-Score nach OSNet + Detail-Re-Ranking |
| match_visual_score | reine OSNet-Ähnlichkeit |
| match_detail_score | Detail-Ähnlichkeit, 0.5 = neutral |
| match_detail_weight | Gewicht der Detail-Schicht |
| match_reason | stärkste Detail-Einflüsse |
| details_label | erkannte Detailzustände im aktuellen Crop |
| detail_reliability | durchschnittliche Zuverlässigkeit der Detail-Signale |

## Empfohlene erste Werte

```text
detail_weight = 0.12 bis 0.18
detail_min_confidence = 0.50 bis 0.60
```

Für Tests kann `detail_weight` höher gestellt werden. Für reale Stabilität sollte es niedrig bleiben, weil die aktuelle Version noch heuristisch arbeitet.

## Wichtige Interpretation

Ein Detail mit Score `0.50` ist neutral.

```text
0.80 = ähnlich und leicht positiver Einfluss
0.50 = neutral / unbekannt / nicht vergleichbar
0.20 = widersprüchlich und leicht negativer Einfluss
```

Volatile Details wie Kappe, Kapuze, Uhr, Rucksack oder Shirt-Aufdruck werden nicht als harte Identitätsmerkmale betrachtet.
