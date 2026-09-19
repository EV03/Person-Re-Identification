# G1-G4-Auswertung aller ReID-Presets

Stand: 19. September 2026

## Szenarien und Videoablage

Die Videos bleiben lokale Testdaten und werden nicht automatisch in Git
aufgenommen. Der Versuchsstarter erwartet folgende Ordner:

```text
Test-daten/
├── G1/
│   └── g1_vor_zurueck_drehen.mp4
├── G2/
│   └── g2_raus_und_rein.mp4
├── G3/
│   └── g3_aehnliche_kleidung_raus_rein.mp4
└── G4/
    └── g4_kreuzen_umeinander_drehen.mp4
```

Es werden exakt vier Videos verwendet, je genau eines pro Gruppe:

- **G1:** Beide Personen bewegen sich vor und zurück und drehen sich. Es gibt
  keine weitere Interaktion und keine Kreuzung.
- **G2:** Beide Personen laufen innerhalb desselben Videos aus dem Bild und
  wieder hinein.
- **G3:** Beide Personen tragen ähnliche Kleidung und laufen innerhalb
  desselben Videos aus dem Bild und wieder hinein.
- **G4:** Beide Personen kreuzen sich und drehen sich stark im Kreis
  umeinander.

Jedes Video enthält zwei reale Personen und wird vollständig als eine Sequenz
ausgewertet. Verschiedene Presets und Wiederholungen erhalten neue, isolierte
Datenbanken. Der Manifest-Generator bricht mit einer verständlichen Meldung ab,
wenn eine Gruppe fehlt oder mehr als ein Video enthält.

## Manifest erzeugen

```powershell
python scripts/evaluate_g1_g4_suite.py init-manifest `
  --video-root Test-daten `
  --output data/g1_g4_manifest.csv
```

Das CSV enthält genau vier Zeilen: Gruppe, Sequenz, Videopfad, erwartete reale
Personenzahl und Notizen. `expected_person_count` steht für alle Videos auf 2.

## Alle Presets ausführen

```powershell
python scripts/evaluate_g1_g4_suite.py run `
  --manifest data/g1_g4_manifest.csv `
  --output-dir data/evaluation_suites `
  --max-frames 0
```

Ohne `--modes` laufen:

- B0, A1, A2 und A3 aus der Main-Pipeline;
- D1 als vollständige Details-Tracking-Pipeline;
- D2-D6 als Ablationen jeweils einer D1-Methode.

Ein eingeschränkter Lauf ist zum Beispiel möglich mit:

```powershell
python scripts/evaluate_g1_g4_suite.py run `
  --manifest data/g1_g4_manifest.csv `
  --modes default details_tracking details_no_reranking `
  --device cpu
```

Für OSNet muss ein gültiger Checkpoint im Preset stehen oder mit
`--checkpoint models/reid/<datei>.pth` angegeben werden.

## Erzeugte Berichte

Jeder Suite-Lauf erzeugt unter `data/evaluation_suites/g1_g4_<lauf-id>/`:

- `report.md`: gemeinsamer Vergleich aller Presets und Gruppen;
- `comparison.json` und `comparison.csv`: sämtliche Einzelläufe;
- `aggregate.csv`: Mittelwerte je Preset und Gruppe;
- `profiles/<preset>.md/.json/.csv`: separate Berichte je Preset;
- `units/`: isolierte Laufmanifeste, `frames.jsonl`, MOT-Exporte und Videos.

Der Bericht kombiniert zwei Metrikfamilien für **jede** Pipeline:

1. Main-Laufmetriken: Frames, Verarbeitungszeit, FPS, Real-Time-Factor und
   Artefaktpfade aus dem technischen Laufmanifest.
2. Identitätsdiagnostik aus `frames.jsonl`: Zuordnungsquote, Zahl der Track- und
   Personen-IDs, Profilzahl gegenüber der erwarteten Personenzahl,
   Track→Person-Ausgabewechsel, Person→Track-Fragmentierung sowie
   Strong/Weak/Low- und Pending-Zähler.

Diese zweite Familie wird identisch auf B0/A1/A2/A3 und D1-D6 angewendet. Bei
Main-Presets bleiben D-spezifische Zonen erwartungsgemäß null.

## Aussagegrenze ohne Ground Truth

Die Videos allein enthalten keine realen Identitätslabels oder Referenzboxen.
Darum sind Profilzahl, Ausgabewechsel und Fragmentierung zunächst
**Output-Diagnosen**. Sie dürfen nicht als echte ID-Switches, IDF1, MOTA,
Precision oder Recall bezeichnet werden. Für solche Genauigkeitsmetriken müssen
pro Frame Ground-Truth-Boxen mit stabilen realen Personen-IDs annotiert und über
einen separaten Ground-Truth-Vergleich ausgewertet werden.
