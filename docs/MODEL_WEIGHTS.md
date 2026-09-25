# OSNet: veröffentlichte ReID-Gewichte, kein eigenes Training

B0/A2 verwenden `osnet_x1_0`, trainiert auf **MSMT17 (`combineall=True`)**.
Wir haben das Modell weder selbst trainiert noch feinabgestimmt. Die Wahl ist
ein festgehaltener Ausgangspunkt, keine Behauptung optimaler Genauigkeit.

Quelle: [offizieller Torchreid Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO),
Abschnitt "MSMT17 (combineall=True) -> Market1501 & DukeMTMC-reID", Zeile OSNet-x1.0.
Dieser Checkpoint ist ReID-trainiert, nicht der separat verlinkte ImageNet-Checkpoint.

- Lokaler Pfad: `data/models/osnet_x1_0_msmt17.pth`
- Download: [Originaldatei der Autoren](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view)
- Originalname: `osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`
- SHA-256: `48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727`
- Architektur/Ausgabe: OSNet-x1.0, 512 Dimensionen, normalisiert
- Eingabe: BGR-Crop -> RGB, Resize 256x128, Torchreid-ImageNet-Normalisierung

## Einrichten

```powershell
python -m pip install -r requirements-evaluation-lock.txt
New-Item -ItemType Directory -Force data/models
python -m gdown 'https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x' -O data/models/osnet_x1_0_msmt17.pth
python -m app.evaluation.smoke --device cpu
```

Die Gewichte wurden lokal heruntergeladen und mit einem echten Encoder-Smoke-Test
geprüft. Sie sind als große Binärdatei von Git ausgeschlossen. Der Standardpfad
prüft die dokumentierte Prüfsumme. Alternative Modelle/Checkpoints lassen sich im
gemeinsamen UI-Editor oder über `--checkpoint`/`--reid-model` an der normalen CLI wählen.
Bei Alternativen Herkunft und Trainingsdaten separat dokumentieren; das Manifest
enthält deren tatsächliche Prüfsumme, aber kann unbekannte Trainingsdaten nicht erraten.

Fehlende Checkpoints führen zu einem Fehler; es gibt keinen stillen ImageNet-Fallback.
Tensor-only Laden (`weights_only=True`) und vollständige Backbone-Prüfung verhindern
versehentliche Teilinitialisierung. Die Klassifikationsschicht für 4101 Trainings-IDs
wird absichtlich nicht übernommen: Sie wird für die Embedding-Ausgabe nicht benötigt.

## Evaluation und Grenzen

Der Checkpoint wurde mit allen MSMT17-Bildern trainiert. Deshalb **MSMT17 nicht als
unabhängigen Testdatensatz verwenden**. Eigene Szenen und ein separat ausgewähltes,
nicht aus diesen Trainingsaufnahmen stammendes Video sind ein Cross-Domain-Test.
Ein öffentlich zugängliches Video ist nicht automatisch frei lizenziert: Quelle,
Lizenz/Nutzungsberechtigung und Annotation in euren Unterlagen festhalten.

Der technische Modelltest belegt Laden/Dimension/Normalisierung, nicht erfolgreiche
Wiedererkennung. Die Referenzschwelle 0,75 muss anhand separater Pilotclips beurteilt
werden. Testclips nicht nachträglich zur Wahl besserer Schwellen verwenden.
