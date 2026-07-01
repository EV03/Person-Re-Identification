# Max Accuracy ReID Variante

Diese Variante ergänzt das bisherige Minimal-Setup um einen ressourcenintensiven Modus:

- `max_accuracy_qdrant_botsort`
- YOLO: `yolov8x.pt` als größtes stabiles YOLOv8-Modell im bisherigen Projektkontext
- Tracker: `botsort.yaml`
- Vector Store: Qdrant statt reinem SQLite-Cosine-Scan
- ReID Encoder: `torchreid` / OSNet
- eigene Direction-Erkennung: deaktiviert, sobald BoT-SORT aktiv ist
- größere Inferenzauflösung: `image_size=1280`
- ReID-Prüfung: jedes Frame (`reid_every_n_frames=1`)
- initiales Quality-Gate: mindestens 5 gute Crops vor erstem ReID-Match

## Qdrant ohne Docker verwenden

Wenn auf dem Rechner kein Docker läuft, nutzt der Max-Modus standardmäßig **Qdrant Local**:

```env
QDRANT_MODE=local
QDRANT_LOCAL_PATH=data/qdrant_local_max_accuracy
QDRANT_COLLECTION=person_reid_max_accuracy
```

Dabei läuft kein Qdrant-Server auf `localhost:6333`. Der `qdrant-client` öffnet lokal einen persistenten Datenordner im Python-Prozess. Für euren lokalen MVP ist das die einfachste Variante.

Prüfen:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install qdrant-client
python scripts/check_qdrant_setup.py
python -m streamlit run app/ui/streamlit_app.py
```

In der UI:

```text
Vector store: qdrant
Qdrant mode: local
Qdrant local path: data/qdrant_local_max_accuracy
```

## Qdrant mit Server/Docker verwenden

Nur nötig, wenn wirklich ein separater Qdrant-Server laufen soll:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

Dann in der UI:

```text
Vector store: qdrant
Qdrant mode: server
Qdrant URL: http://localhost:6333
```

Standardwerte für Serverbetrieb:

```env
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=person_reid_max_accuracy
QDRANT_PREFER_GRPC=false
```

## Abhängigkeiten installieren

```bash
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Falls `torchreid` bereits funktioniert, ist der zweite Befehl nicht nötig.

## Streamlit starten

```bash
python -m streamlit run app/ui/streamlit_app.py
```

Dann im Modus-Dropdown auswählen:

```text
Max Accuracy ReID (max_accuracy_qdrant_botsort)
```

## CLI-Beispiel ohne Docker

```bash
python -m app.main \
  --source data/input/testvideo.mp4 \
  --mode max_accuracy_qdrant_botsort \
  --store qdrant \
  --qdrant-mode local \
  --qdrant-local-path data/qdrant_local_max_accuracy \
  --qdrant-collection person_reid_max_accuracy \
  --tracker botsort.yaml \
  --model yolov8x.pt
```

## CLI-Beispiel mit Qdrant-Server

```bash
python -m app.main \
  --source data/input/testvideo.mp4 \
  --mode max_accuracy_qdrant_botsort \
  --store qdrant \
  --qdrant-mode server \
  --qdrant-url http://localhost:6333 \
  --qdrant-collection person_reid_max_accuracy \
  --tracker botsort.yaml \
  --model yolov8x.pt
```

## Optional: neueres/größeres Ultralytics-Modell

Der Modus nutzt bewusst `yolov8x.pt`, weil euer bisheriger Stand von `yolov8n.pt` ausgeht und `yolov8x.pt` die direkte große Variante davon ist. Wenn eure installierte Ultralytics-Version neuere Modelle unterstützt, kann das Modell ohne Codeänderung überschrieben werden:

```env
REID_MAX_YOLO_MODEL=yolo11x.pt
```

Alternativ im Streamlit-Sidebar-Feld `YOLO model` überschreiben.

## Technische Änderung gegenüber dem Minimal-Modus

Vorher:

```text
YOLOv8n + ByteTrack + Torchreid/OSNet + SQLite Vector Store + eigene Direction-Erkennung
```

Jetzt im Max-Modus:

```text
YOLOv8x + BoT-SORT + Torchreid/OSNet + Qdrant Vector Store + interne Direction-Erkennung aus
```

SQLite bleibt intern als leichtes Metadaten-/Event-Log aktiv, damit die bestehenden Tabellen in Streamlit weiter funktionieren. Die eigentliche Embedding-Suche läuft im Qdrant-Backend.

## Qdrant Troubleshooting

Use the same Python executable for installation and Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install qdrant-client
python -c "import sys; print(sys.executable); import qdrant_client; print(qdrant_client.__version__)"
```

Wenn kein Docker läuft, ist das korrekt:

```text
Qdrant mode: local
```

Wenn die UI im Servermodus sagt, dass Qdrant nicht erreichbar ist, dann läuft kein Qdrant-Server. Entweder Server starten oder zurück auf `Qdrant mode: local` stellen.
