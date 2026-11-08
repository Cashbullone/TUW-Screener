# TUW-Screener (Turtle-Strategie auf Wochenkerzen)

Eigenständiger Screener, der die Regeln der Turtle-Händler (Richard Dennis, William Eckhardt) auf **Wochenkerzen** von US-Aktien anwendet.
Du prüfst einmal pro Woche am Freitagabend und setzt die Orders am Montag. GitHub holt jeden Freitag nach US-Börsenschluss die Kurse von
**S&P 500 und Nasdaq-100**, bildet Wochenkerzen (Montag bis Freitag), rechnet die Turtle-Kennzahlen und veröffentlicht sie zusammen mit der App als Webseite.

Was die App zeigt:

- **Orderzettel für Montag:** die Stop-Orders für die kommende Woche (Verkaufs-Stops, Zukäufe, neue Einstiege) mit Marke und Stückzahl.
- **Depot:** deine Positionen, nachgespielt Woche für Woche. Die App meldet Stop und Ausstieg, die nächste Zukauf-Marke und das Risiko bis zum Stop.
- **Live-Kurse im Depot (optional):** Mit einem kostenlosen Finnhub-Schlüssel (Feld unter „Kursdaten“, bleibt nur im Browser) zeigt das Depot Kurs, Wert und Gewinn in Echtzeit und warnt, wenn der Live-Kurs den Verkaufs-Stop erreicht oder die Zukauf-Marke berührt. Der Orderzettel bleibt bei den Wochenkerzen.
- **Neue Einstiege:** Aktien, die kurz vor dem Ausbruch über das 20- oder 55-Wochen-Hoch stehen, mit Limits (je Aktie, Sektor, Richtung).
- **In der letzten Woche ausgelöste Ausbrüche:** was in der abgeschlossenen Woche ausgelöst wurde, samt Skip-Regel von System 1.

## Die Regeln in Kürze

| Baustein | Regel |
| --- | --- |
| N | Wilder-Durchschnitt der True Range über 20 Wochen |
| Einheit | Stück = Depot × Risiko ÷ N (Original: 1 % Risiko, hier Standard 0,25 %) |
| System 1 | Einstieg über dem 20-Wochen-Hoch, Ausstieg unter dem 10-Wochen-Tief. Nach einem Gewinner-Trade wird der nächste Ausbruch übersprungen |
| System 2 | Einstieg über dem 55-Wochen-Hoch, Ausstieg unter dem 20-Wochen-Tief, ohne Überspringen |
| Stop | 2 N unter dem letzten Kaufkurs |
| Zukauf | alle ½ N über dem letzten Kaufkurs, bis zu 4 Einheiten je Aktie |
| Limits | 4 Einheiten je Markt, 6 je Sektor, 12 je Richtung |

**Rhythmus:** Die laufende Woche zählt nie. Am Freitagabend (ab etwa 23:30 Uhr österreichischer Zeit) steht die abgeschlossene Woche in den Daten.
Am Montag legst du die Stop-Orders aus dem Orderzettel für die ganze Woche in den Markt. Ein Ausbruch am Dienstag wird also am Dienstag gefüllt.
Stops und Ausstiege stehen auf Wochenbasis: N ist größer als beim Tages-Screener (TU), die Stops sitzen weiter weg, die Einheiten fallen kleiner aus
und es gibt weniger Signale.

Die Rechenwege, alle Annahmen und die nicht bestätigten Punkte stehen in der App unter „So rechnet die App“.

## Einrichtung (einmalig)

1. **Veröffentlichung einschalten:** im Repository „Settings“ → „Pages“ → bei „Source“ **GitHub Actions** wählen.
2. **Ersten Lauf starten:** Reiter „Actions“ → „Kurse aktualisieren und Seite veröffentlichen“ → „Run workflow“.
   Der erste Lauf dauert etwa 10 bis 25 Minuten. Danach läuft die Automatik jeden Freitag um 22:30 UTC, als Rückfall am Samstag um 06:15 UTC.
3. **App öffnen:** `https://cashbullone.github.io/TUW-Screener/`.

## Was wo gespeichert wird

- Im Repository liegen nur App, Skript und Tests. Die Kursdaten entstehen bei jedem Lauf neu und gehen direkt auf die Webseite.
- Dein Depot und deine Einstellungen bleiben **nur in deinem Browser** (eigener Speicher, getrennt vom TU-Screener). Auf einem zweiten Gerät
  übertragst du sie mit „Depot exportieren“ und „Depot importieren“.
- Die Webseite ist öffentlich erreichbar. Die Kursdaten stammen von Yahoo Finance und sind für den privaten Gebrauch gedacht. Gib den Link nicht weiter.

## Einstellungen der Automatik

In `update.yml` unter „Kursdaten holen“ kannst du die auskommentierten Zeilen aktivieren: `MIN_PRICE`, `MIN_AVG_VOL` (Durchschnitt der letzten 50 Handelstage), `INDEXES`.
Den Sektor liest das Skript aus den Indexlisten, sonst aus `universe/sectors.csv` und zuletzt von Yahoo.

## Was das Skript an die App liefert

Für jede Aktie: N, die Marken der letzten 10, 20 und 55 Wochen, die in der letzten Woche ausgelösten Ausbrüche, den Zustand der Skip-Regel
(aus einer Nachspielung auf Wochenbasis) und die Reihe der letzten 104 Wochenkerzen (Hoch, Tief, Schluss, N) für die Depotprüfung.

## Tests

`python tests/test_core.py` (Rechenkern), `python tests/test_assemble.py` (Aufbereitung mit künstlichen Daten ohne Netzwerk),
`python tests/make_fake.py ORDNER` erzeugt eine künstliche `prices.json` zum Ausprobieren der App.

## Wenn etwas nicht klappt

- **Lauf schlägt fehl:** den Lauf unter „Actions“ öffnen und die Meldung bei „Kursdaten holen“ ansehen. Bei weniger als 300 Aktien bricht der Lauf absichtlich ab, die Seite zeigt dann weiter die letzten Daten.
- **Yahoo blockiert Abrufe:** einfach „Run workflow“ erneut starten.
- **Börsenfeiertag am Freitag:** Die Woche erscheint erst in den Daten, wenn die nächste Woche begonnen hat.
- **Nach 60 Tagen ohne Änderung im Repository** pausiert GitHub geplante Läufe. Dann unter „Actions“ wieder aktivieren.

Das ist ein Regelwerk aus der Literatur und keine Anlageberatung. Die App macht keinen Backtest.
