# Ergebnisse der Vier-Video-Evaluation

> **Überholter Evaluationsstand:** Diese Läufe verwendeten noch einen Abstand
> von drei Frames zwischen Initialkandidaten. Nach dessen Entfernung müssen die
> zwölf Läufe neu erzeugt werden; die folgenden Werte dürfen nicht als Ergebnis
> der aktuellen Implementierung übernommen werden.

Stand: 20. September 2026. Der Lauf `four_groups_20260920_134315_99af698e`
verarbeitete je ein Video aus G1 bis G4 mit B0, A2 und A3. Alle zwölf isolierten
Läufe wurden vollständig abgeschlossen. Jede Gruppe und Variante begann mit
leerem Personenbestand und frischem Tracker.

## Konfiguration

B0 verwendete OSNet-x1.0 mit MSMT17-Gewichten, YOLOv8n und ByteTrack. Matching-
und Update-Ähnlichkeit betrugen 0,75. Fünf Initialkandidaten wurden im Abstand von
drei Frames gesammelt; bekannte Spuren wurden alle fünf Frames für ein Update
geprüft. Die Qualitätsgrenzen lagen bei 0,55 für Kandidaten und 0,65 für Updates,
die Schärfegrenzen bei 0,40 beziehungsweise 0,45 bei Randkontakt und die
Mindestbewertung des Seitenverhältnisses bei 0,50. Die Überlappungsgrenze betrug
0,15 mit elf Frames Abkühlzeit.

A2 setzte ausschließlich die fünf Qualitäts- und Initialgrenzen auf null. A3
übernahm B0 und deaktivierte ausschließlich den Update-Ähnlichkeitsschutz mit
`min_update_similarity = -1`.

## Ereignisergebnisse

| Kenngröße | B0 | A2 | A3 |
|---|---:|---:|---:|
| Rückkehr korrekt | 5/6 (83,3 %) | 3/6 (50,0 %) | 5/6 (83,3 %) |
| G2: Rückkehr korrekt | 3/3 | 2/3 | 3/3 |
| G3: Rückkehr korrekt | 2/3 | 1/3 | 2/3 |
| Falsche bestehende ID bei Rückkehr | 0/6 | 0/6 | 0/6 |
| Neue ID für registrierte Person | 1/6 | 3/6 | 1/6 |
| G1: unbekannter Eintritt innerhalb von 2 s | 0/1 | 1/1 | 0/1 |
| G4: Neuzuweisung innerhalb von 2 s korrekt | 0/2 | 1/2 | 1/2 |
| Median bis zur korrekten Rückkehr-ID | 1,13 s | 1,13 s | 1,13 s |

Ein weiterer Eintritt in G2 wurde ausgeschlossen, weil das Video vor Ablauf des
vollständigen Zwei-Sekunden-Fensters endet und zunächst nur eine randberührende
Teilansicht sichtbar ist.

## Profilupdates im G4-Fenster

Im festgelegten Fenster von Frame 301 bis 391 erreichten drei Kandidaten die
Update-Entscheidung. Alle drei zeigten den Eigentümer des zugeordneten Profils.
B0 lehnte sie mit Ähnlichkeiten von 0,620 bis 0,644 ab, A2 mit 0,551 bis 0,573.
A3 nahm alle drei Eigenkandidaten an. Fremdkandidaten traten nicht auf; die
Schutzwirkung der Schwelle gegen Profilverunreinigung ist deshalb mit diesem
Bestand nicht messbar.

## Laufzeit

| Variante | Frames | Verarbeitung | FPS | Real-Time-Faktor |
|---|---:|---:|---:|---:|
| B0 | 4.165 | 217,95 s | 19,11 | 1,569 |
| A2 | 4.165 | 221,49 s | 18,80 | 1,594 |
| A3 | 4.165 | 218,94 s | 19,02 | 1,576 |

Die vier Videos umfassen zusammen etwa 138,9 Sekunden bei ungefähr 30 FPS. Die
Messung erfolgte unter Windows 11 mit Python 3.13.1 und PyTorch 2.12.0 auf der CPU;
CUDA war nicht verfügbar. Die Verarbeitung erreichte daher in keiner Variante die
Eingangsbildrate. Pro Video und Variante wurde nur ein Lauf gemessen.

## Provenienz und Aussagegrenzen

Die Ereignisreferenz steht in `docs/evaluation/four_group_event_reference.json`.
Sie wurde anhand der Originalvideos, annotierten Ausgaben und Frameprotokolle
manuell geprüft. Das Auswertungsskript schreibt daraus `evaluated_event_log.csv`,
`evaluated_update_log.csv` und `evaluation_results.json` in das Batch-Verzeichnis.
Der Eingabe-ZIP-Hash lautet
`171a85b7c4db448ad569026c95c06a734a6c17175c69d39dd158f91453187de3`.
Die Laufmanifeste halten zusätzlich Konfiguration, Modell- und Dateihashes,
Umgebung sowie den Quellcodezustand fest.

Die Untersuchung umfasst zwei Personen, vier kurze Eigenaufnahmen und sechs
Rückkehrereignisse. Die Aufnahmen wurden bereits während der Entwicklung betrachtet,
es gibt keine unabhängige Zweitannotation und keine Laufzeitwiederholungen. Die
Ergebnisse sind deshalb deskriptiv und nicht auf andere Personen, Kameras oder
Aufnahmebedingungen verallgemeinerbar.
