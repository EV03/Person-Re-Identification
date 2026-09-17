# Projekt-Exposé

Dieses Verzeichnis enthält das LaTeX-Exposé zum Person-Re-Identification-Projekt.
Aufbau, KOMA-Script-Stil, Deckblatt und modulare Dateistruktur orientieren sich
an der HSRM-Vorlage:

https://gitlab.cs.hs-rm.de/jklam001/abschlussarbeit-latex-template

Vor einer Abgabe sind in `preamble/header.tex` mindestens Name,
Matrikelnummer, Studiengang und Betreuung zu ergänzen.

## Kompilieren

Mit Tectonic aus diesem Verzeichnis:

```powershell
New-Item -ItemType Directory -Force build
tectonic expose.tex --outdir build
```

Mit einer vollständigen TeX-Live-/MiKTeX-Installation:

```powershell
New-Item -ItemType Directory -Force build
latexmk -xelatex -interaction=nonstopmode -outdir=build expose.tex
```
