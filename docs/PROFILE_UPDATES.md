# Personenprofile: Qualitätsprüfung, Akkumulation und Update-Schutz

## Zwei unabhängige Probleme

Modellwechsel werden durch getrennte Datenbanken pro Encoder-Konfiguration
behandelt. Falsche Tracker-Zuordnungen werden durch eine zusätzliche
Ähnlichkeitsschwelle vor dem Profilupdate teilweise abgefangen. Die Schwelle
ist kein Ersatz für einen guten Tracker und repariert keine Personenkennung.

## Qualitätswert Q eines Crops

Alle Teilwerte werden zwischen 0 und 1 bewertet. Es handelt sich um feste
Heuristiken, nicht um eine trainierte Wahrscheinlichkeit korrekter Identität.

| Teilwert | Berechnung | Gewicht |
|---|---|---|
| Schärfe S | `clip(Varianz(Laplace(Graubild)) / 150, 0, 1)` | 0,25 |
| Helligkeit H | `clip(1 - abs(mittlerer Grauwert - 128) / 128, 0, 1)` | 0,15 |
| Größe G | `min(1, Cropbreite/(2*min_crop_width), Crophöhe/(2*min_crop_height))`; im Nenner mindestens 1 | 0,25 |
| Seitenverhältnis A | Höhe/Breite: linear von 0 auf 1 bei 1,1–1,5; 1 bei 1,5–4; linear von 1 auf 0 bei 4–5; sonst 0 | 0,15 |
| Randkontakt R | 0,65, wenn die ursprüngliche Box innerhalb von 3 Pixeln an mindestens einem Bildrand liegt; sonst 1 | 0,10 |
| Detektionskonfidenz D | YOLO-Konfidenz, auf 0–1 begrenzt | 0,10 |

`Q = 0.25*S + 0.15*H + 0.25*G + 0.15*A + 0.10*R + 0.10*D`.
Die Mindestgröße wird vorher als harte Bedingung geprüft; für Q wird die
geclippte/gepaddete Cropgröße verwendet. Schärfe und Größe haben das höchste Gewicht.

Die Standard-Kandidatenschwelle ist 0,55, die zusätzliche Updateschwelle 0,65.
Eine Updateschwelle von 0,65 bedeutet **nicht** 65 % Identitätssicherheit.
Neue UI-Presets dürfen die Schwellen verändern; die Teilwertkonstanten bleiben fest.

## Erste fünf Crops und weitere Beobachtungen

Für jeden akzeptierten Crop liefert der Encoder einen normalisierten Vektor z.
Das Gewicht ist `w = max(Q, 0.05)`; der Boden bleibt bei A2 ohne Qualitätsschwellen erhalten.
Ein unbekannter Track sammelt standardmäßig fünf akzeptierte Beobachtungen.
Zwischen zwei akzeptierten Initialkandidaten liegen mindestens drei Frames;
ungeeignete Frames verschieben die nächste Annahme nicht künstlich nach hinten.

```text
S = Summe(wi*zi), i=1..5
W = Summe(wi), i=1..5
p = normalize(S / W)
```

`p` dient zur Suche nach einem vorhandenen Profil. Ein neues Profil speichert
S als `embedding_sum`, W als `embedding_weight_sum` und p als `mean_embedding`.
`observations` zählt akzeptierte Einzel-Crops, also zunächst fünf, nicht ein Batch.
Der beste Crop liefert den Snapshot, aber nicht mehr das Gesamtgewicht des Batches.

Für einen bekannten Track wird standardmäßig im Videoframe 10, 20, 30 usw.
der aktuelle Crop geprüft; es wird nicht der beste Crop aus zehn Frames gewählt.
Er muss Mindestgrößen und beide Qualitätsschwellen erfüllen, bevor der Encoder
aufgerufen wird. Der Qualitätswert wird einmal berechnet; effektiv gilt die
höhere Grenze aus `min_embedding_quality` und `min_update_quality`.
Neue, noch nicht zugeordnete Tracks benötigen nur die Kandidatenqualität. Ihr
eigener Abstand `initial_candidate_every_n_frames` ist vom Updateintervall
bekannter Tracks unabhängig. Zusätzlich gelten zwei separat einstellbare
Schärfegrenzen: `min_initial_blur_score` für jeden Initialkandidaten und die
strengere Grenze `min_border_blur_score`, wenn die Bounding-Box einen Bildrand
berührt. Die Standardwerte sind 0,40 und 0,45. Abgelehnte Bilder füllen den
Initialpuffer nicht; der unbekannte Track wartet weiter auf bessere Crops.

Vor der Cropqualität greift eine separate Überlappungssperre. Für jedes Paar
erkannter Personen wird die Schnittfläche relativ zur kleineren Bounding-Box
berechnet. Liegt sie standardmäßig über `max_person_overlap_ratio = 0,15`,
werden beide Ausschnitte weder codiert noch zur Profilanlage oder zum Update
verwendet. Danach bleibt der Track standardmäßig weitere
`overlap_cooldown_frames = 10` Frames gesperrt. Ein noch nicht zugeordneter
Track verliert dabei seine zuvor gesammelten Initialcrops, damit Beobachtungen
vor und nach einer möglichen Trackverwechslung nicht vermischt werden. Der
Frame-Export weist diese Fälle als `overlapping_person` beziehungsweise
`overlap_cooldown` aus und enthält `person_overlap_ratio`.

Das Embedding eines für ein Update zugelassenen Crops wird anschließend
mit dem **bisherigen** Personenprofil verglichen:

```text
similarity = cosine(p, z_neu)
akzeptieren, falls similarity >= min_update_similarity
```

Der vorläufige Default ist 0,82, unabhängig vom konfigurierbaren
`match_threshold` für die erste Zuordnung. Das ist kein empirisch optimaler Wert:
auf Pilotclips einstellen, danach einfrieren. -1 lässt alle gültigen Ähnlichkeiten zu.

Nur bei Annahme:

```text
S_neu = S + w_neu*z_neu
W_neu = W + w_neu
p_neu = normalize(S_neu / W_neu)
```

Ein neuer Track, der ein vorhandenes Profil wiederfindet, ergänzt entsprechend
die vollständige Fünf-Crop-Summe, ebenfalls nach der Ähnlichkeitsprüfung.
Die Aktualisierung bleibt hinter `ProfileUpdater`; Speicher enthält keine Vektorarithmetik.

Die Summe wird mit Float64-Präzision gespeichert. Eine nachträgliche Änderung
der Beobachtungsreihenfolge verändert den exakten gewichteten Mittelwert nicht,
abgesehen von Rundungsabweichungen. **Bei aktiviertem Ähnlichkeitsschutz** können
verschiedene Reihenfolgen jedoch verschiedene Beobachtungen zulassen; diese
online Annahmeentscheidung ist weiterhin vom bisherigen Profil abhängig.

## Ablehnung und Grenzen

Bei zu geringer Ähnlichkeit bleiben Embedding, Summe, Gewicht, Beobachtungszahl,
Profilzeitpunkte und bester Snapshot unverändert. Ein Diagnoseereignis
`profile_update_rejected` speichert den Versuch; ein etwaiger Kandidaten-Snapshot
ist nicht der neue Profilsnapshot. Frame-Exportfelder:

- `profile_update_accepted`
- `update_similarity`
- `profile_update_reason`

Der Track behält seine bisherige Personenkennung, auch nach Ablehnung. Es gibt
keine automatische Neuzuordnung. Ähnlich aussehende Fremdpersonen können die
Schwelle bestehen; initiale falsche Zuordnungen bleiben ebenfalls möglich.
Ein Perspektivwechsel kann ein eigentlich korrektes Update ablehnen.

## Datenbanken pro Encoder und Versuch

UI und normale CLI verwenden für geteilte Läufe:

`data/db/encoders/<encoder_key>/reid.sqlite3`.

Der Schlüssel umfasst Backend, Architektur, Checkpoint-Inhalt und den
Vorverarbeitungsvertrag (bei Torchreid auch relevante Paketversionen).
Farbhistogramme ignorieren ungenutzte OSNet-Einstellungen. Gleiche Gewichte
unter anderem Dateinamen erhalten denselben Schlüssel; geänderte Gewichte
einen anderen. Wieder zum ursprünglichen Encoder wechseln verwendet seinen
bisherigen Bestand. Der Pfad wird in der UI bzw. normalen CLI angezeigt.

Isolierte Versuche legen zusätzlich eine neue Einheit an; dort liegt die DB
unter `data/experiments/<unit_id>/db/encoders/<encoder_key>/reid.sqlite3`.
Varianten/Wiederholungen bleiben getrennt, auch mit demselben Encoder.
Registrierung und Rückkehr innerhalb einer Einheit teilen Profile, nicht Tracker.

Die Standardpipeline löst auch direkt übergebene Pfade encodergebunden auf,
sofern kein eigener Profilspeicher/-manager injiziert wird. Die Auflösung ist
idempotent. Explizit injizierte Encoder/Repositories müssen im eigenen
Einstiegspunkt passend zusammengesetzt werden; Standard-CLI, UI und
Evaluationsstarter übernehmen die Zuordnung automatisch.

## Bestehende Daten und Evaluation

Alte DB-Dateien werden nicht gelöscht oder automatisch übernommen. Das neue
Profilformat verwendet einen eigenen Namespace. Bestehende normalisierte
Altprofile ohne rohe Summe lassen sich nicht exakt rekonstruieren: Bei einem
Update in einem ausdrücklich wiederverwendeten Altbestand gibt es eine
verständliche Fehlermeldung, statt einer erfundenen Summe.

Die neue Methode gilt konsistent für B0/A1/A2/A3. A3 deaktiviert nur den
Ähnlichkeitsschutz mit `min_update_similarity=-1`, nicht die Profilupdates.
A2 entfernt nur Qualitätsgrenzen,
nicht den Ähnlichkeitsschutz oder die Gewichtung. Code, Parameter und Gewichte
vor finalen Tests einfrieren. Der Vergleich mit deaktiviertem Update-Schutz
kann über ein Pilotpreset mit `min_update_similarity=-1` erfolgen; das deaktiviert
nicht die Updates selbst. Profilupdates an/aus sind aktuell kein eigenes Preset.
