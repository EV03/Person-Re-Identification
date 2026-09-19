"""Generate the vector architecture figure used by the LNCS paper."""

from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_WIDTH = 360
PAGE_HEIGHT = 205

INK = (0.12, 0.12, 0.12)
MID = (0.42, 0.42, 0.42)
LIGHT = (0.96, 0.96, 0.96)
ACCENT = (0.90, 0.90, 0.90)


def set_color(pdf: canvas.Canvas, color: tuple[float, float, float]) -> None:
    pdf.setStrokeColorRGB(*color)
    pdf.setFillColorRGB(*color)


def draw_box(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    *,
    fill: tuple[float, float, float] = LIGHT,
    font_size: float = 7.7,
    bold_first: bool = False,
) -> None:
    pdf.setLineWidth(0.75)
    pdf.setStrokeColorRGB(*INK)
    pdf.setFillColorRGB(*fill)
    pdf.roundRect(x, y, width, height, 3.5, stroke=1, fill=1)

    line_height = font_size + 2.0
    total_height = line_height * len(lines)
    baseline = y + (height + total_height) / 2 - line_height + 1
    for index, line in enumerate(lines):
        font_name = "Helvetica-Bold" if bold_first and index == 0 else "Helvetica"
        pdf.setFont(font_name, font_size)
        pdf.setFillColorRGB(*INK)
        text_width = stringWidth(line, font_name, font_size)
        pdf.drawString(x + (width - text_width) / 2, baseline - index * line_height, line)


def arrow_head(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    dx: float,
    dy: float,
    size: float = 4.0,
) -> None:
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    path = pdf.beginPath()
    path.moveTo(x, y)
    path.lineTo(x - size * ux + size * 0.55 * px, y - size * uy + size * 0.55 * py)
    path.lineTo(x - size * ux - size * 0.55 * px, y - size * uy - size * 0.55 * py)
    path.close()
    pdf.drawPath(path, stroke=0, fill=1)


def draw_arrow(
    pdf: canvas.Canvas,
    points: list[tuple[float, float]],
    *,
    both_ends: bool = False,
) -> None:
    pdf.setLineWidth(0.9)
    set_color(pdf, INK)
    path = pdf.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    pdf.drawPath(path, stroke=1, fill=0)

    end_dx = points[-1][0] - points[-2][0]
    end_dy = points[-1][1] - points[-2][1]
    arrow_head(pdf, *points[-1], end_dx, end_dy)
    if both_ends:
        start_dx = points[0][0] - points[1][0]
        start_dy = points[0][1] - points[1][1]
        arrow_head(pdf, *points[0], start_dx, start_dy)


def draw_lane_label(pdf: canvas.Canvas, text: str, x: float, y: float) -> None:
    pdf.setFont("Helvetica-Bold", 6.2)
    pdf.setFillColorRGB(*MID)
    pdf.drawString(x, y, text)


def generate(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    pdf.setTitle("Verarbeitungskette des lokalen ReID-Prototyps")
    pdf.setAuthor("Person Re-Identification Masterprojekt")

    draw_lane_label(pdf, "TRACKING", 12, 193)
    draw_box(pdf, 12, 151, 72, 32, ["Video oder", "Webcam"])
    draw_box(pdf, 101, 151, 78, 32, ["Personendetektion", "YOLOv8n"], bold_first=True)
    draw_box(pdf, 196, 151, 70, 32, ["Spurzuordnung", "ByteTrack"], bold_first=True)
    draw_box(pdf, 283, 151, 65, 32, ["Box und", "Track-ID"])

    draw_arrow(pdf, [(84, 167), (101, 167)])
    draw_arrow(pdf, [(179, 167), (196, 167)])
    draw_arrow(pdf, [(266, 167), (283, 167)])

    pdf.setStrokeColorRGB(0.78, 0.78, 0.78)
    pdf.setLineWidth(0.55)
    pdf.line(12, 137, 348, 137)
    draw_lane_label(pdf, "RE-IDENTIFIKATION UND PROFILVERWALTUNG", 12, 127)

    draw_box(pdf, 276, 84, 72, 34, ["Personen-", "ausschnitt"])
    draw_box(pdf, 190, 84, 70, 34, ["Größen- und", "Qualitätsprüfung"])
    draw_box(pdf, 105, 84, 69, 34, ["ReID-Encoder", "OSNet / Histogramm"], bold_first=True, font_size=7.2)
    draw_box(
        pdf,
        12,
        78,
        77,
        46,
        ["Profilentscheidung", "Matching oder", "geschütztes Update"],
        fill=ACCENT,
        font_size=7.0,
        bold_first=True,
    )

    draw_arrow(pdf, [(315.5, 151), (315.5, 118)])
    draw_arrow(pdf, [(276, 101), (260, 101)])
    draw_arrow(pdf, [(190, 101), (174, 101)])
    draw_arrow(pdf, [(105, 101), (89, 101)])

    draw_box(pdf, 12, 15, 77, 32, ["Synthetische", "Personen-ID"])
    draw_box(pdf, 125, 15, 102, 32, ["Encodergebundener", "Profilspeicher"], font_size=7.4)

    draw_arrow(pdf, [(50.5, 78), (50.5, 47)])
    draw_arrow(pdf, [(75, 78), (75, 56), (176, 56), (176, 47)], both_ends=True)

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    generate(repository_root / "docs" / "figures" / "systemarchitektur.pdf")
