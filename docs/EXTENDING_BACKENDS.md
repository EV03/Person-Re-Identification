# Tracker, Matching und Profilverfahren austauschen

Typisierte Verträge trennen Implementierungen. Standard sind Cosine Matching,
exakte gewichtete Akkumulation und eine zusätzliche Ähnlichkeitsprüfung vor
Profilupdates. Die Rechnung ist in [PROFILE_UPDATES.md](PROFILE_UPDATES.md) erklärt.

## Zuständigkeiten

```text
PersonReIdPipeline
  ├─ PersonTracker             → Detection mit Box und Track-ID
  ├─ EmbeddingEncoder          → normalisiertes Embedding
  ├─ ProfileManager / ProfileService
  │    ├─ ProfileMatcher       → passende Personenkennung oder kein Treffer
  │    ├─ ProfileUpdater       → neues Profil aus Beobachtung und altem Profil
  │    └─ ProfileRepository    → Profile und Ereignisse speichern/laden
  └─ RunRepository             → Laufstatus und Metadaten speichern
```

SQLite implementiert die beiden Speicherverträge. Der **Profilspeicher** ist
einfach die Ablage der Personen-Embeddings und Metadaten, kein zusätzliches
Modell. Ihr müsst SQLite nicht ersetzen. Die Trennung erlaubt, ein anderes
Matching zu untersuchen, ohne SQL oder die Pipeline umzuschreiben.

## Einen eigenen Tracker einsetzen

Vertrag: `app/pipeline/contracts.py`, `PersonTracker.track_frame`.
Eine Basisklasse ist nicht nötig; ein typisiertes Python-Protocol beschreibt
die erforderliche Methode.

```python
import numpy as np
from app.config import PipelineConfig
from app.pipeline.orchestrator import PersonReIdPipeline
from app.storage.models import Detection

class MyTracker:
    def track_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        # Eigenen Detektor/Tracker aufrufen und dessen Ergebnisse übersetzen.
        # Hier nur ein leeres Adaptergerüst, keine Trackerimplementierung:
        return []

def my_tracker_factory(config: PipelineConfig) -> MyTracker:
    return MyTracker()

pipeline = PersonReIdPipeline(
    config=PipelineConfig(),
    tracker_factory=my_tracker_factory,
)
# pipeline.process("data/input/pilot.mp4")
```

Alternativ eine bereits erzeugte Instanz mit `tracker=MyTracker()` übergeben.
Die Factory wird erst innerhalb von `process()` aufgerufen, damit Ladefehler
im Laufmanifest landen. Pro Quelle eine neue Pipeline und Trackerinstanz
erzeugen; nicht dieselbe zustandsbehaftete Instanz über Videos hinweg teilen.

Adapterregeln:

- Unveränderten BGR-Eingabeframe lesen, nicht überschreiben.
- Nur Personen liefern; Klassenfilterung liegt beim Adapter.
- Bounding Boxes als Pixel-XYXY, Konfidenz im Bereich 0 bis 1.
- Stabile ganzzahlige Track-IDs innerhalb einer Quelle, fehlende IDs als `None`.
- Optional `release()` für Ressourcen; die Pipeline ruft es auch bei Fehlern auf.
- Optional `describe_backend()` mit JSON-kompatiblen Einstellungen und
  Modellreferenzen/Hashes. Implementierungsname und Beschreibung landen im Manifest.

`models` enthält die vom tatsächlichen Adapter beschriebenen Modelle;
`configured_models` bewahrt die Referenzen aus den Konfigurationsfeldern.
Bei eigenen Adaptern ohne Beschreibung bleiben tatsächliche Modellreferenzen
explizit unbekannt, statt versehentlich YOLO-/OSNet-Defaults zu behaupten.

Der Standardadapter bleibt `UltralyticsPersonTracker`. Dessen `tracker`-Setting
wählt weiterhin ByteTrack/BoT-SORT bzw. ein YAML. Eigene Backends werden aktuell
im Python-Einstiegspunkt zusammengesetzt, nicht über einen neuen UI-Auswahlschalter.

Für den Versuchsstarter denselben Aufbau als Pipelinefactory verwenden:

```python
from app.evaluation.runner import run_unit

def make_pipeline(config, paths):
    return PersonReIdPipeline(config, paths, tracker_factory=my_tracker_factory)

# run_unit(sources=[...], config=..., pipeline_factory=make_pipeline)
```

## Matching oder Profilupdate einsetzen

Die Verträge und vollständigen Eingaben stehen in `app/reid/repository.py`.
Die Standardimplementierungen in `app/reid/operations.py` enthalten keine
Datenbank- oder Modellaufrufe und lassen sich separat testen.

```python
pipeline = PersonReIdPipeline(
    config=config,
    matcher=my_matcher,     # implementiert ProfileMatcher.match(...)
    updater=my_updater,     # implementiert ProfileUpdater.update(...)
)
```

Der Matcher erhält Query, Profile, Schwelle und ausgeschlossene Personen-IDs.
Er muss sichtbare/reservierte IDs berücksichtigen und liefert `MatchResult`
oder `None`. Der Updater erhält das bisherige Profil, Embedding, Gewicht,
Snapshotinformationen und Zeitpunkt; er liefert ein neues `IdentityProfile`.
Bei einer Initialisierung/Rückkehr erhält er zusätzlich ein `EmbeddingBatch`
mit roher Summe, Gewichtssumme und Anzahl der Einzel-Crops. `ProfileManager`
liefert bei einem Update eine `ProfileUpdateDecision`; die Pipeline exportiert sie.
Speichern übernimmt anschließend der Service, nicht die Policy.

Auch ein kompletter `profiles=...`-Manager ist möglich. Dann dessen Matcher und
Updater am Manager konfigurieren; eine widersprüchliche doppelte Konfiguration
lehnt die Pipeline ab.

`WeightedMeanProfileUpdater` bewahrt die rohe Summe in `IdentityProfile.embedding_sum`.
Alte Profile ohne Akkumulationszustand werden nicht erfunden nachgebildet.
Neue eigene Policies müssen den Batch-Vertrag berücksichtigen. Beim vollständigen
eigenen Profilmanager liegt auch die Update-Schutzentscheidung in dessen Verantwortung.

## Anderen Speicher einsetzen

```python
from app.reid.service import ProfileService

profiles = ProfileService(my_repository, matcher=my_matcher, updater=my_updater)
pipeline = PersonReIdPipeline(config=config, profiles=profiles)
```

`my_repository` implementiert `ProfileRepository`; Laufdaten bleiben hier in
der Standard-SQLite-Ablage. Alternativ `store=...` für einen Adapter übergeben,
der sowohl ProfileRepository als auch RunRepository erfüllt.

`save_observation()` muss Profil und Ereignis atomar speichern. Der aktuelle
Service setzt sequenzielle Zugriffe voraus: keine parallelen Evaluationsworker
auf demselben Bestand. Für kleine Versuche bleibt die lineare Suche geeignet.
Andere Encodergewichte benötigen auch bei gleicher Dimension einen neuen Bestand.
Die Standardpipeline löst auch explizite Pfade mit `paths_for_encoder(paths, config)`
aus `app/storage/encoder_paths.py` automatisch auf. Bei injizierten Stores liegt
die Namespace-Wahl im eigenen Einstiegspunkt. Standard-UI, CLI und
Evaluationsstarter tun dies automatisch. Eigene Encoder mit abweichender
Vorverarbeitung benötigen eine eigene passende Namespace-Definition.

`SQLiteVectorStore.search()` und `add_or_update_person()` bleiben als
Kompatibilitätsaufrufe erhalten, delegieren aber ausschließlich an den Service.
Neue Anwendungspfade verwenden den Service direkt, nicht diese Fassaden.

## Konfiguration erweitern

Neue gemeinsame Parameter genau einmal in `PipelineSettings` (`app/config.py`)
definieren. `ModeConfig` und `PipelineConfig` erben diese Felder; bestehende
flache Preset-JSONs und Keyword-Aufrufe bleiben kompatibel.
Beide Konfigurationen sind unveränderlich; für Änderungen
`dataclasses.replace(config, match_threshold=...)` verwenden.
Gemeinsame Defaultwerte einschließlich Umgebungsvariablen gelten jetzt auch
für Presets, nicht nur für direkt erstellte Laufkonfigurationen.

Bei einem neuen Laufzeitfeld zusätzlich ein UI-Widget ergänzen. Die Tests prüfen
gemeinsame Felder, JSON-Roundtrip, UI-Abdeckung, unabhängige Tracker, alternative
Policies, Speicherwechsel und atomare Speicherung.
