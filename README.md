_*Worum es im Projekt geht*_
-

In dem Projekt wollen wir ein System aufbauen, das Personen in Videosegmenten wiedererkennen kann.
Also nicht nur in einem einzelnen Bild, sondern auch dann, wenn sich Personen durch die Szene bewegen,
sich kreuzen, kurz verschwinden oder später wieder auftauchen. Uns geht es dabei vor allem um die
praktische Frage, welche Ansätze unter realistischen Bedingungen gut funktionieren und wie man sie
sinnvoll kombinieren kann.
Mit „Real-Time“ ist dabei keine harte Echtzeit im technischen Sinn gemeint, sondern eine möglichst
geringe Latenz und eine Verarbeitung, die so nah wie möglich an Echtzeit abläuft. Ziel ist es, dass das
System die Kamerabilder ohne spürbare Verzögerung verarbeitet und Ergebnisse liefert, auch wenn
einzelne Verarbeitungsschritte je nach Hardware und Modellwahl unterschiedlich schnell ablaufen
können.

_*Wie wir das angehen wollen*_
-

Geplant ist eine modulare Pipeline, die aus mehreren Bausteinen besteht. Zuerst sollen Personen im
Video erkannt werden. Danach sollen diese Personen über mehrere Frames hinweg verfolgt werden,
damit Bewegungen innerhalb einer Szene stabil abgebildet werden können. Wenn Personen kurz verloren
gehen oder später erneut auftauchen, soll über Re-Identification geprüft werden, ob es sich um dieselbe
Person handelt.
Als Eingang ist dabei eine Kameraaufnahme beziehungsweise ein Live-Videostream vorgesehen, aus dem
die einzelnen Frames an das System übergeben und dort weiterverarbeitet werden.
Zusätzlich wollen wir uns anschauen, wie man das Video sinnvoll in Abschnitte unterteilen oder mit Key
Frames arbeiten kann, damit nicht jeder Frame mit demselben Aufwand verarbeitet werden muss. Je
nach Fortschritt des Projekts können auch weitere Bausteine dazukommen, zum Beispiel
Segmentierungsansätze, falls diese bei schwierigen Szenen einen Mehrwert bringen.

_*Verwendete Technologien und Tools*_
-


_*Ziel*_

Am Ende soll eine funktionsfähige und erweiterbare Pipeline entstehen, mit der Personen in
Videosegmenten erkannt, verfolgt und wiedererkannt werden können. Genauso wichtig ist aber auch die
Auswertung: Wir wollen nachvollziehbar zeigen, welche Ansätze in der Praxis gut funktionieren, wo
Probleme auftreten und welche Kombinationen aus Qualität und Effizienz am sinnvollsten sind.
Dabei geht es ausdrücklich um einen Proof of Concept beziehungsweise einen prototypischen Aufbau und
nicht um ein vollständiges Produkt.

System overview: 
                ┌──────────────────────────┐
                │      Live Kamera        │
                │ (USB / RTSP Stream)     │
                └──────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │   Frame Preprocessing    │
                │ (Resize / Normalize)     │
                └──────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │   Person Detection       │
                │      (YOLOv8)            │
                │ → Bounding Boxes         │
                └──────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Multi-Object Tracking    │
                │ (ByteTrack / OC-SORT)    │
                │ → Track IDs (temporär)   │
                └──────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Key-Frame Selection      │
                │ (Quality / Sharpness /   │
                │  Occlusion / Diversity)  │
                └──────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Re-Identification Model  │
                │ (OSNet / FastReID)       │
                │ → Embeddings (vectors)   │
                └──────────┬──────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ Similarity Matching      │
                │ (Cosine Similarity)      │
                │ → Identity Decision      │
                └──────────┬──────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌────────────────────┐         ┌────────────────────┐
│ Known Person Match │         │ New Person Entry   │
│ → Global ID        │         │ → New Embedding    │
└────────────────────┘         └────────────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │  Gallery / Memory DB     │
                │ (Embeddings + IDs)       │
                └──────────────────────────┘
