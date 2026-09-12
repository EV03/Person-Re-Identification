# Qualitätsfilterung im dokumentierten ReID-Kern

Die Qualitätsprüfung soll ungeeignete Person-Crops vor dem Encoding und vor
Profilupdates zurückhalten. Ihr empirischer Nutzen ist noch zu evaluieren.

## Implementierter Ablauf

- Geometrisch ungültige und zu kleine Crops werden verworfen.
- Schärfe und Größe tragen jeweils 0,25 zum Qualitätswert bei; Helligkeit und
  Seitenverhältnis jeweils 0,15; Randkontakt und Detektionskonfidenz jeweils 0,10.
- Neue Tracks sammeln standardmäßig drei Crops mit Qualität mindestens 0,55.
- Die normalisierten Kandidaten werden qualitätsgewichtet kombiniert.
- Der beste Crop dient als Snapshot.
- Bekannte Tracks werden alle fünf Frames auf ein Profilupdate geprüft.
- Updates benötigen mindestens Qualität 0,65 und werden qualitätsgewichtet gespeichert.

Bei unbekannten Tracks wird pro Frame nach brauchbaren Crops gesucht; das
Fünf-Frame-Intervall betrifft die bereits zugeordneten Tracks.
Crops stammen immer aus dem unveränderten Analysebild. Boxen und Texte werden
nur auf die Ausgabekopie gezeichnet.

## Vergleich laut Evaluationsplan

B0 verwendet OSNet mit den genannten Schwellen, A1 ein Farbhistogramm.
A2 setzt beide Annahmeschwellen auf null, behält Mindestgröße, Qualitätsgewichtung,
Initialpuffer und Snapshot-Auswahl bei. Die Gewichtung verwendet im Code ein
Mindestgewicht von 0,05.

Die Modellgewichte für B0/A2 müssen noch ausdrücklich festgelegt werden.
Die drei Presets allein ersetzen keinen Experiment-Runner mit getrennten Datenbeständen.

## Grenzen

Ein guter Crop kann zur falschen Person gehören. Die aktuelle Update-Regel
überprüft die Identitätszuordnung nicht erneut. Qualitätsfilterung ist deshalb
kein vollständiger Schutz vor Tracker-ID-Switches. Auch ein einzelnes gemitteltes
Profil kann unterschiedliche Ansichten nur begrenzt repräsentieren.

Vor weiteren Heuristiken sollen ein vollständiger Vorhersageexport und eine
Pilot-Auswertung entstehen. Details stehen in [EVALUATION_SCOPE.md](EVALUATION_SCOPE.md).
